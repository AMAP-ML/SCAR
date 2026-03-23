import pdb

from datasets import load_from_disk
import io
import numpy as np
import PIL
from PIL import Image
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# PIPE dataset
class PIPE_Dataset(Dataset):
    def __init__(self,
                 args,
                 dataset_path,
                 llm_tokenizer,
                 mode='train',
                 ):

        self.args = args
        # PIPE Dataset path
        self.dataset_path = load_from_disk(dataset_path)

        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer

    def __len__(self,):
        return len(self.dataset_path)

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[
            :, :, np.newaxis
        ] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, PIL.Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        tensor_img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float()
        )  # (Channels, Height, Width) format.

        # Add batch dimension.
        return tensor_img

    def __getitem__(self, index):
        # IMG_START = self.llm_tokenizer.vocab.boi_id
        # IMG_END = self.llm_tokenizer.vocab.eoi_id
        # Loading Path...
        data = self.dataset_path[index]
        # ['source_img', 'target_img', 'Instruction_VLM-LLM', 'Instruction_Class', 'Instruction_Ref_Dataset', 'object_location', 'target_img_dataset', 'img_id', 'ann_id']

        # convert into torch style
        if random.random() > 0.5:
            input_img = data['source_img']
            edited_img = data['target_img']
            input_img = Image.open(io.BytesIO(input_img['bytes'])).convert('RGB')
            edited_img = Image.open(io.BytesIO(edited_img['bytes'])).convert('RGB')
            # edit_txt = data['Instruction_VLM-LLM'] + data['object_location']
            edit_txt = data['Instruction_VLM-LLM']

        else:
            input_img = data['target_img']
            edited_img = data['source_img']
            input_img = Image.open(io.BytesIO(input_img['bytes'])).convert('RGB')
            edited_img = Image.open(io.BytesIO(edited_img['bytes'])).convert('RGB')
            # edit_txt = 'Remove ' + data['Instruction_VLM-LLM'][4:] + data['object_location']
            edit_txt = 'Remove ' + data['Instruction_VLM-LLM'][4:]


        edit_text_tokens_and_mask = self.llm_tokenizer(
            edit_txt,
            max_length=120,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        edit_txt_token = edit_text_tokens_and_mask['input_ids']
        edit_txt_attn_mask = edit_text_tokens_and_mask['attention_mask']

        input_ids = edit_txt_token[0]
        input_ids_attn_mask = edit_txt_attn_mask[0]

        input_img = self._whiten_transparency(input_img)
        input_img = self._vqgan_input_from(input_img)
        edited_img = self._whiten_transparency(edited_img)
        edited_img = self._vqgan_input_from(edited_img)

        # _input_img = (input_img.permute(1,2,0)+1)/2 * 255.
        # _input_img.save(f"{self.args.gpt_ckpt[:-3]}/pipe_{index:08d}_input.png")
        # _input_img.save(f"pipe_{index:08d}_input.png")
        # _edited_img = (edited_img.permute(1,2,0)+1)/2 * 255.
        # _edited_img.save(f"{self.args.gpt_ckpt[:-3]}/pipe_{index:08d}_edit.png")
        # _edited_img.save(f"pipe_{index:08d}_edit.png")
        # # Create the content to be saved
        # content = f"Edited Text:\n{edit_txt}"
        # # Save the content to the file
        # with open(f'{self.args.gpt_ckpt[:-3]}/pipe_{index:08d}_txt.txt', "w") as file:
        #     file.write(content)

        return {
                'index': index,
                'dataset': 'pipe',
                'mode': 1,
                'input_ids': input_ids,
                'input_ids_attn_mask': input_ids_attn_mask,
                # 'target_ids': target_ids,
                'input_img': input_img,
                'edited_img': edited_img,
                # '_input_ids': _input_ids,
                # '_input_ids_attn_mask': _input_ids_attn_mask,
                # '_target_ids': _target_ids,
                }
