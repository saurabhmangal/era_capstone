# +
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from lightning.pytorch import LightningDataModule
from transformers import AutoProcessor, AutoTokenizer
from torch.utils.data import random_split, DataLoader
import requests

class llavadataset(Dataset):
    def __init__(self, coco_data, phi_model_name, clip_model_name,train_flag,tokenizer):

        self.tokenizer  = tokenizer
        self.processor  = AutoProcessor.from_pretrained(clip_model_name)
        self.caption_dataset = coco_data

        train_size = int(0.9 * len(self.caption_dataset))
        print(f"Train size {train_size} and validation size {len(self.caption_dataset) - train_size}")

        if train_flag == 'train':
          self.caption_dataset = self.caption_dataset[0:train_size]
        else:
          self.caption_dataset = self.caption_dataset[train_size:]

      
    def __len__(self):
        return len(self.caption_dataset)

    def __getitem__(self, idx):

        # from image perspective
        img_url = self.caption_dataset[idx][2]
        caption = self.caption_dataset[idx][3]

        # image load
        image_load = Image.open(requests.get(img_url,stream=True).raw)
        image_processed = self.processor(images=image_load, return_tensors="pt") ['pixel_values']
        image_processed = image_processed.squeeze(0)
        a = self.tokenizer(caption, return_tensors="pt", return_attention_mask=False)
        return(image_processed , a['input_ids'].squeeze(0))
  


def collate_fn(batch):
    image_embeddings, captions = zip(*batch)
    image_embeddings_stacked = torch.stack(image_embeddings, dim=0)
    captions_padded = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=50256)
    return (image_embeddings_stacked, captions_padded)
