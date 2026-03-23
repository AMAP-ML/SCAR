# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List, Any
import pdb
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
from .dinov2_utils.DINOv2 import vit_base
import random
from .gpt_edit_dino import LabelEmbedder, CaptionEmbedder, MLP, RMSNorm, FeedForward, KVCache, Attention, TransformerBlock, CrossAttentionAdapter
from .gpt_edit_dino import precompute_freqs_cis, precompute_freqs_cis_2d_edit, apply_rotary_emb, precompute_freqs_cis_2d_edit_anyshape
# from feature_encoders.build import Semantic_Encoder

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'
    model_mode: str = None
    distill_mode: str = None

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    block_lr_size: int = 64
    max_batch_size: int = 32
    max_seq_len: int = 2048
    dino_dim: int = 1536
    args: Any = None


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_mode = config.model_mode
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.block_lr_size = config.block_lr_size
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.args = config.args
        self.dino_dim = config.dino_dim
        self.dino_downsample = getattr(config.args, 'dino_downsample', 1)
        # self.dino_loss_level = getattr(config.args, 'dino_loss_level', 4)
        if self.model_type == 'edit':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        elif self.model_type == 'control':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.register_buffer("img_uncond_embedding", nn.Parameter(torch.randn(self.block_size, config.dim) / config.dim ** 0.5))
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))


        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size

        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        # alignment layer
        if self.args.use_vq_distill: 
            self.alignment_layer = nn.Conv2d(config.dim, self.dino_dim//4, 2, 2)

        if self.args.dino_level == "4th":
            self.downsample_conv = nn.Conv2d(self.dino_dim, self.dino_dim, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample_conv = nn.Conv2d(self.dino_dim//4, self.dino_dim//4, kernel_size=3, stride=2, padding=1)
        if self.args.dino_level == "4th":
            self.upsampler = nn.Sequential(
                    nn.Conv2d(self.dino_dim, self.dino_dim * 4, kernel_size=3, padding=1),  # 生成 4x 通道（2x upscale）
                    nn.PixelShuffle(upscale_factor=2),
                    nn.Conv2d(self.dino_dim, self.dino_dim, kernel_size=3, padding=1)       # refine
                ) 
        else:
            self.upsampler = nn.Sequential(
                    nn.Conv2d(self.dino_dim//4, self.dino_dim, kernel_size=3, padding=1),  # 生成 4x 通道（2x upscale）
                    nn.PixelShuffle(upscale_factor=2),
                    nn.Conv2d(self.dino_dim//4, self.dino_dim//4, kernel_size=3, padding=1)       # refine
                ) 

        self.initialize_weights()
        self.pretrained_dinov2 = './autoregressive/models/dinov2_utils/dinov2_vitb14_pretrain.pth'
        self.init_DINOv2()
        if hasattr(self.args, 'cond_img_size'):
            self.cond_img_size = self.args.cond_img_size
        else:
            self.cond_img_size = 448
        self.prefilling_size = (self.cond_img_size//14)**2
        self.prefilling_token = self.cond_img_size//14 // self.dino_downsample
        # self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        self.freqs_cis = precompute_freqs_cis_2d_edit_anyshape(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, self.prefilling_token)
        if self.args.dino_level == "1th":
            self.dino_mlp =  nn.Linear(3072//4, config.dim, bias=False)
        else:
            self.dino_mlp =  nn.Linear(3072, config.dim, bias=False)

    def init_DINOv2(self):
        self.dinov2 = vit_base()
        print(f'load model from: {self.pretrained_dinov2}')
        pretrain_dict = torch.load(self.pretrained_dinov2)
        msg = self.dinov2.load_state_dict(pretrain_dict, strict=True)
        n_parameters = sum(p.numel() for p in self.dinov2.parameters())
        print('Missing keys: {}'.format(msg.missing_keys))
        print('Unexpected keys: {}'.format(msg.unexpected_keys))
        print(f"=> loaded successfully '{self.pretrained_dinov2}'")
        print('DINOv2 Count: {:.5f}M'.format(n_parameters / 1e6))


    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype,input_downsample_ratio=1):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d_edit_anyshape(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, self.prefilling_token)


    def preprocess_cond(self, image, img_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(image, (img_size, img_size), mode=mode, align_corners=False)

    def forward(
        self, 
        input_txt_embs: torch.Tensor,
        input_img: torch.Tensor,
        edited_img: Optional[torch.Tensor] = None,
        edited_img_indices: Optional[torch.Tensor] = None,
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        input_mode: Optional[torch.Tensor] = None,
    ):
        if self.model_mode == None:
            if (input_img is not None) and (edited_img_indices is not None):
                # input ids
                input_txt_embeddings = self.cls_embedding(input_txt_embs, train=self.training)[:,:self.cls_token_num]
                input_txt_embeddings = self.tok_dropout(input_txt_embeddings)

                input_img_dino_ipts = self.preprocess_cond(input_img, self.cond_img_size)
                # pdb.set_trace()
                with torch.no_grad():
                    input_img_dinov2_outs = self.dinov2(input_img_dino_ipts, is_training=True) # [ns,h*w,c]
                    source_dinov2_outs_last4 = input_img_dinov2_outs["x_norm_patchtokens_mid4"].contiguous()
                    source_dinov2_outs_last1 = input_img_dinov2_outs["x_norm_patchtokens"].contiguous()
                if self.args.dino_level == "1th":
                    cond_dinov2_outs_last4_hr = source_dinov2_outs_last1
                else:
                    cond_dinov2_outs_last4_hr = source_dinov2_outs_last4
                n_ = cond_dinov2_outs_last4_hr.shape[0]
                h_ =  self.cond_img_size//14
                w_ =  self.cond_img_size//14
                cond_dinov2_outs_last4_hr = rearrange(cond_dinov2_outs_last4_hr.view(n_, h_, w_, -1), 'n h w c -> n c h w').contiguous()
                cond_dinov2_outs_last4_lr = self.preprocess_cond(cond_dinov2_outs_last4_hr, h_//self.dino_downsample)
                cond_dinov2_outs_last4_ds = self.downsample_conv(cond_dinov2_outs_last4_hr)
                cond_dinov2_outs_last4 = cond_dinov2_outs_last4_ds + cond_dinov2_outs_last4_lr

                similarity_loss = self.args.sample_loss_weight * F.mse_loss(self.upsampler(cond_dinov2_outs_last4),cond_dinov2_outs_last4_hr)

                cond_dinov2_outs_last4 = rearrange(cond_dinov2_outs_last4.view(n_, -1, h_//self.dino_downsample, w_//self.dino_downsample), 'n c h w -> n (h w) c').contiguous()
                cond_dinov2_outs_last4 = self.dino_mlp(cond_dinov2_outs_last4)
                if self.training and self.args.use_distill:
                    edited_img_dino_ipts = self.preprocess_cond(edited_img, self.cond_img_size)
                    with torch.no_grad():
                        edited_img_dinov2_outs = self.dinov2(edited_img_dino_ipts, is_training=True) # [ns,h*w,c]
                        edited_dinov2_outs_last4 = edited_img_dinov2_outs["x_norm_patchtokens_mid4"].contiguous()
                        edited_dinov2_outs_last1 = edited_img_dinov2_outs["x_norm_patchtokens"].contiguous()
                    if self.args.dino_level == "1th":
                        edited_dinov2_outs_last4 = edited_dinov2_outs_last1
                    edited_dinov2_outs_last4 = rearrange(edited_dinov2_outs_last4.view(n_, h_, w_, -1), 'n h w c -> n c h w').contiguous()
                    edited_dinov2_outs_last4_lr = self.preprocess_cond(edited_dinov2_outs_last4, h_//self.dino_downsample)
                    edited_dinov2_outs_last4_ds = self.downsample_conv(edited_dinov2_outs_last4)
                    edited_dinov2_outs_last4 = edited_dinov2_outs_last4_ds + edited_dinov2_outs_last4_lr

                    edited_dinov2_outs_last4 = rearrange(edited_dinov2_outs_last4.view(n_, -1, h_//self.dino_downsample, w_//self.dino_downsample), 'n c h w -> n (h w) c').contiguous()
                    edited_dinov2_outs_last4 = self.dino_mlp(edited_dinov2_outs_last4)

                edited_img_embeddings = self.tok_embeddings(edited_img_indices)
                token_embeddings = torch.cat((cond_dinov2_outs_last4, input_txt_embeddings, edited_img_embeddings), dim=1)[:, :-1]
                self.freqs_cis = self.freqs_cis.to(token_embeddings.device)
            
                targets = edited_img_indices

        # ############## joint embedding dropout #################
        if self.model_mode == 'joint_cls_emb':
            if (input_img is not None) and (edited_img_indices is not None):
                force_drop_ids = torch.rand(input_txt_embs.shape[0], device=input_txt_embs.device)
            else:
                if input_img is not None: # prefill in inference
                    input_img_dino_ipts = self.preprocess_cond(input_img, self.cond_img_size)
                    with torch.no_grad():
                        input_img_dinov2_outs = self.dinov2(input_img_dino_ipts, is_training=True) # [ns,h*w,c]
                        source_dinov2_outs_last4 = input_img_dinov2_outs["x_norm_patchtokens_mid4"].contiguous()
                        source_dinov2_outs_last1 = input_img_dinov2_outs["x_norm_patchtokens"].contiguous()
                    if self.args.dino_level == "1th":
                        cond_dinov2_outs_last4_hr = source_dinov2_outs_last1
                    else:
                        cond_dinov2_outs_last4_hr = source_dinov2_outs_last4
                    n_ = cond_dinov2_outs_last4_hr.shape[0]
                    h_ =  self.cond_img_size//14
                    w_ =  self.cond_img_size//14
                    cond_dinov2_outs_last4_hr = rearrange(cond_dinov2_outs_last4_hr.view(n_, h_, w_, -1), 'n h w c -> n c h w').contiguous()
                    cond_dinov2_outs_last4_lr = self.preprocess_cond(cond_dinov2_outs_last4_hr, h_//self.dino_downsample)
                    cond_dinov2_outs_last4_ds = self.downsample_conv(cond_dinov2_outs_last4_hr)
                    cond_dinov2_outs_last4 = cond_dinov2_outs_last4_ds + cond_dinov2_outs_last4_lr
                    cond_dinov2_outs_last4 = rearrange(cond_dinov2_outs_last4.view(n_, -1, h_//self.dino_downsample, w_//self.dino_downsample), 'n c h w -> n (h w) c').contiguous()
                    cond_dinov2_outs_last4 = self.dino_mlp(cond_dinov2_outs_last4)

                    input_txt_embeddings = self.cls_embedding(input_txt_embs, train=self.training)[:,:self.cls_token_num]                
                    token_embeddings = torch.cat((cond_dinov2_outs_last4, input_txt_embeddings), dim=1)
                else: 
                    token_embeddings = self.tok_embeddings(edited_img_indices)

                mask = self.causal_mask[:, None, input_pos]

        h = token_embeddings
        learning_loss = 0

        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        for i, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, input_pos, mask)
        h = self.norm(h)

        # semantic features
        features = None
        distill_loss = 0.
        distill_loss+= similarity_loss
        if self.training:
            if self.args.use_distill:
                dino_size = edited_dinov2_outs_last4.shape[1]
                features_dino = h[:, :dino_size].clone()        
                distill_loss_dino = F.mse_loss(features_dino, edited_dinov2_outs_last4)
                distill_loss+=self.args.distill_loss_weight*distill_loss_dino

            if self.args.use_vq_distill:
                features_ori = h[:, -self.block_size:].clone()
                img_size = self.args.image_size//self.args.downsample_size
                features_ori = features_ori.view(n_, img_size, img_size, -1)
                features_ori = rearrange(features_ori, 'n h w c -> n c h w').contiguous()
                features = self.alignment_layer(features_ori)
                features = rearrange(features, 'n c h w -> n (h w) c').contiguous()
                with torch.no_grad():
                    edited_img_dino_ipts1 = self.preprocess_cond(edited_img, self.args.image_size//self.args.downsample_size*7)
                    edited_img_dinov2_outs1 = self.dinov2(edited_img_dino_ipts1, is_training=True) # [ns,h*w,c]
                    edited_dinov2_outs_last1_1 = edited_img_dinov2_outs1["x_norm_patchtokens"].contiguous()

                distill_loss_vq = F.mse_loss(features, edited_dinov2_outs_last1_1)
                distill_loss += self.args.distill_vq_loss_weight*distill_loss_dino


        logits = self.output(h).float()
        
        if self.training:
            logits = logits[:, self.prefilling_size//(self.dino_downsample*self.dino_downsample) + self.cls_token_num - 1:].contiguous()
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if self.training:
            return logits, loss, learning_loss, distill_loss
        return logits, loss, distill_loss
        


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}