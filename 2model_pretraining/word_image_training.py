#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pickle
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model_network import CLIPPhi2Model, train_model
from dataset import collate_fn, llavadataset

# Proxy setup, if necessary
try:
    os.environ['HTTP_PROXY'] = 'http://185.46.212.90:80'
    os.environ['HTTPS_PROXY'] = 'http://185.46.212.90:80'
    print("Proxy exported")
except Exception as e:
    print("Could not set proxy:", e)

# Ensure CUDA is available, otherwise fall back to CPU
if torch.cuda.is_available():
    print(f"Using CUDA: {torch.cuda.device_count()} GPUs available")
    device = torch.device('cuda')
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device('cpu')

# Load your dataset
with open("coco_dataset_pickle", "rb") as fp:
    coco_unpickle = pickle.load(fp)

# Tokenizer and model setup
clip_model_name = "openai/clip-vit-base-patch32"
phi_model_name = "microsoft/phi-2"
train_batch_size = 4
val_batch_size = 2
tokenizer = AutoTokenizer.from_pretrained(phi_model_name, trust_remote_code=True, use_cache=True)
tokenizer.save_pretrained("saved_tokenizer")

# Model initialization and DataParallel wrapping
MModalGPT = CLIPPhi2Model()
if torch.cuda.is_available():
    MModalGPT = torch.nn.DataParallel(MModalGPT).to(device)

# Data loaders setup
train_dataloader = DataLoader(
    llavadataset(coco_unpickle, phi_model_name, clip_model_name, 'train', tokenizer),
    collate_fn=collate_fn, batch_size=train_batch_size, num_workers=20, shuffle=True, pin_memory=True)

val_dataloader = DataLoader(
    llavadataset(coco_unpickle, phi_model_name, clip_model_name, 'val', tokenizer),
    collate_fn=collate_fn, batch_size=val_batch_size, num_workers=20, shuffle=True, pin_memory=True)

# Optimizer setup
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, MModalGPT.parameters()), lr=1e-6)

# Set float32_matmul_precision to 'medium'
torch.set_float32_matmul_precision('high')
# torch.set_float32_matmul_precision('highest')

# Train the model
train_model(MModalGPT, train_dataloader, val_dataloader, optimizer, device, max_steps=100000, model_save_step=1000, model_val_step=100, log_step=100, max_token_filter=30, tokenizer=tokenizer)
