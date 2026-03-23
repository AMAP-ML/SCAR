# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from glob import glob
import time
import argparse
import os
import inspect
from natsort import natsorted
import wandb
import yaml
from types import SimpleNamespace
import sys
import pdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.distributed import init_distributed_mode
from utils.logger import create_logger
from dataset.build import build_dataset
from dataset.Edit_ALLinOne import EditALLinOne_Dataset
from autoregressive.models.gpt_edit import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models

from language.t5 import T5Embedder
from feature_encoders.build import Semantic_Encoder

def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank==0 and args.use_wandb:
        wandb.init(
            project="editAR",
            name=os.path.basename(args.output_dir),
        )
        wandb.define_metric("train/*", step_metric="train/global_step")

    # Setup an experiment folder:
    checkpoint_dir = f"{args.output_dir}/checkpoints"
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(args.output_dir)
        logger.info(f"Experiment directory created at {args.output_dir}")

    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # setup tokenizer
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    assert os.path.exists(args.vq_ckpt)
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # Setup model
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        model_mode=args.gpt_mode,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        distill_mode=args.distill_mode,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.use_distill:
        semantic_encoder = Semantic_Encoder(args.distill_mode, precision, device)

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # dataset = build_dataset(args, llm_tokenizer=t5_model.tokenizer)
    dataset = EditALLinOne_Dataset(args, dataset_jsonl_path=args.dataset_jsonl_path, 
                                   dataset_dir_path=args.dataset_dir_path, llm_tokenizer=t5_model.tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    folder_names = natsorted([folder_name for folder_name in os.listdir(checkpoint_dir)])

    if len(folder_names) > 0:
        model_path = os.path.join(checkpoint_dir, folder_names[-1])
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "steps" in checkpoint:
            train_steps = checkpoint["steps"]
        else:
            train_steps = 0
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {model_path}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    elif args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        # # if "optimizer" in checkpoint:
        # #     optimizer.load_state_dict(checkpoint["optimizer"])
        # if "steps" in checkpoint:
        #     train_steps = checkpoint["steps"]
        # else:
        #     train_steps = 0
        # del checkpoint
        # start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        # logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        # logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        # pdb.set_trace()

        train_steps = 0
        start_epoch = 0
    else:
        train_steps = 0
        start_epoch = 0

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_llm_loss = 0
    running_distill_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            # input_img, edited_img, target_ids, input_ids, prompts
            input_img = batch['input_img'].to(device, non_blocking=True)
            edited_img = batch['edited_img'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            input_ids_attn_mask = batch['input_ids_attn_mask'].to(device, non_blocking=True)
            input_mode = batch['mode'].to(device, non_blocking=True)

            # process text ids to embeddings
            with torch.no_grad():
                input_txt_embs = t5_model.model(
                    input_ids=input_ids,
                    attention_mask=input_ids_attn_mask,
                )['last_hidden_state'].detach()
            
            # process image ids to embeddings
            with torch.no_grad():
                _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                _, _, [_, _, edited_img_indices] = vq_model.encode(edited_img)
                input_img_indices = input_img_indices.reshape(input_img.shape[0], -1)
                edited_img_indices = edited_img_indices.reshape(edited_img.shape[0], -1)

            with torch.cuda.amp.autocast(dtype=ptdtype):  
                _, llm_loss, llm_features = model(input_txt_embs=input_txt_embs, input_img_indices=input_img_indices, edited_img_indices=edited_img_indices, input_mode=input_mode)
                if args.use_distill:
                    distill_loss = args.distill_loss_weight * semantic_encoder.compute_distill_loss(edited_img, llm_features)
                    loss = llm_loss + distill_loss
                else:
                    loss = llm_loss
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # Log loss values:
            running_loss += loss.item()
            if args.use_distill:
                running_llm_loss += llm_loss.item()
                running_distill_loss += distill_loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if args.use_distill:
                    avg_llm_loss = torch.tensor(running_llm_loss / log_steps, device=device)
                    dist.all_reduce(avg_llm_loss, op=dist.ReduceOp.SUM)
                    avg_llm_loss = avg_llm_loss.item() / dist.get_world_size()
                    avg_distill_loss = torch.tensor(running_distill_loss / log_steps, device=device)
                    dist.all_reduce(avg_distill_loss, op=dist.ReduceOp.SUM)
                    avg_distill_loss = avg_distill_loss.item() / dist.get_world_size()
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, LLM Loss: {avg_llm_loss:.4f}, Distill Loss: {avg_distill_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                else:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank==0 and args.use_wandb:
                    wandb_dic = {}
                    wandb_dic['train/global_step'] = train_steps
                    wandb_dic['train/loss'] = avg_loss
                    if args.use_distill:
                        wandb_dic['train/llm_loss'] = avg_llm_loss
                        wandb_dic['train/distill_loss'] = avg_distill_loss
                    wandb.log(wandb_dic)
                # Reset monitoring variables:
                running_loss = 0
                running_llm_loss = 0
                running_distill_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                dist.barrier()


    if rank==0:
        # output COMPLETE
        os.makedirs(args.output_dir, exist_ok=True)
        file_path = os.path.join(args.output_dir, 'COMPLETE')
        # Create the empty file
        with open(file_path, 'w') as f:
            pass  # Just create an empty file and close it

    logger.info("Done!")
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--use-distill", action='store_true')
    parser.add_argument("--distill-mode", type=str, choices=['dinov2', 'clip'], default=None)
    parser.add_argument("--distill-loss-weight", type=float, default=0.5, help="distill loss weight")
    parser.add_argument("--use-wandb", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='./pretrained_models/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 'edit'], default="edit")
    parser.add_argument("--gpt-mode", type=str, choices=['img_cls_emb', 'joint_cls_emb'], default=None)
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--output-dir", type=str, default="checkpoints/test")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--dataset_jsonl_path", type=str, default="")
    parser.add_argument("--dataset_dir_path", type=str, default="")


    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_cfg = yaml.safe_load(f)

        yaml_args = SimpleNamespace()
        for key, value in yaml_cfg.items():
            setattr(yaml_args, key, value)

        for key in vars(args):
            cli_value = getattr(args, key)
            default_value = parser.get_default(key)
            if cli_value != default_value:
                setattr(yaml_args, key, cli_value)

        args = yaml_args

    return args


if __name__ == "__main__":

    args = parse_args()
    main(args)
