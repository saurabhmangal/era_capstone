from dataset import llavadataset, collate_fn
import pickle
import peft
from peft import LoraConfig
from transformers import AutoTokenizer,BitsAndBytesConfig, AutoModelForCausalLM, CLIPVisionModel, AutoProcessor
import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd
from torch.nn import functional as F
import csv
import random
from PIL import Image
import requests
import os
from peft import PeftModel
import torch.nn as nn

os.environ['http_proxy'] ='http://abmcpl-blr.it%40adityabirla.com:birla%402019@165.225.104.42'
os.environ['https_proxy'] ='http://abmcpl-blr.it%40adityabirla.com:birla%402019@165.225.104.42'
os.environ['HTTP_PROXY'] ='http://abmcpl-blr.it%40adityabirla.com:birla%402019@165.225.104.42'
os.environ['HTTPS_PROXY']='http://abmcpl-blr.it%40adityabirla.com:birla%402019@165.225.104.42'

clip_model_name = "openai/clip-vit-base-patch32"
phi_model_name  = "microsoft/phi-2"
tokenizer  = AutoTokenizer.from_pretrained(phi_model_name, trust_remote_code=True)
processor  = AutoProcessor.from_pretrained(clip_model_name)
tokenizer.add_tokens('[QA]')
tokenizer.add_special_tokens({'pad_token':'[PAD]'}) 
train_batch_size    = 32
clip_embed = 768
phi_embed  = 2560
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 10
IMAGE_TOKEN_ID = 23893 # token for word comment
max_steps      = 100000
EOS_TOKEN_ID   = 50256
phi_patches    = 49
vocab_size     = 51200
max_generate_length = 100
model_val_step      = 1000
model_log_step      = 100
model_save_step     = 100
torch.set_float32_matmul_precision('medium')
tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer('[QA]')


# training data
csv_file = 'train_token.csv'
qa_dataset = pd.read_csv(csv_file)

# data loaders
train_dataloader = DataLoader(llavadataset(qa_dataset, phi_model_name,clip_model_name,processor),
                  collate_fn=collate_fn, batch_size=train_batch_size, num_workers = num_workers, shuffle=True, pin_memory=True)


file = open('sample_val_data.csv')
csvreader = csv.reader(file)
sample_val_data = []
for row in csvreader:
    sample_val_data.append(row)
print(sample_val_data[1])
file.close()

class SimpleResBlock(nn.Module):
    def __init__(self, phi_embed):
        super().__init__()
        self.pre_norm = nn.LayerNorm(phi_embed)
        self.proj = nn.Sequential(
            nn.Linear(phi_embed, phi_embed),
            nn.GELU(),
            nn.Linear(phi_embed, phi_embed)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

clip_model = CLIPVisionModel.from_pretrained(clip_model_name).to(device)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,)

phi_model = AutoModelForCausalLM.from_pretrained(
    phi_model_name,
    torch_dtype=torch.float32,
    quantization_config=bnb_config,
    trust_remote_code=True
)
phi_model.config.use_cache = False
projection = torch.nn.Linear(clip_embed, phi_embed).to(device)
resblock = SimpleResBlock(phi_embed).to(device)

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        'k_proj',
        'v_proj',
        'fc1',
        'fc2'
    ]
)
peft_model = peft.get_peft_model(phi_model, peft_config).to(device)
peft_model.print_trainable_parameters()

# clip non trainable
for network in [clip_model]:
    for param in network.parameters():
        param.requires_grad_(False)


# In[ ]:


# check trainable paramaeters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"PEFT MODEL:{count_parameters(peft_model)}")
print(f"PROJECTION MODEL:{count_parameters(projection)}")
print(f"CLIP MODEL:{count_parameters(clip_model)}")
print(f"PHI MODEL:{count_parameters(phi_model)}")
print(f"RESNET MODEL:{count_parameters(resblock)}")

if os.path.isfile('model_chkpt/step2_projection.pth'):
    projection.load_state_dict(torch.load('./model_chkpt/step2_projection.pth'))
    resblock.load_state_dict(torch.load('./model_chkpt/step2_resblock.pth'))
    peft_model.from_pretrained(phi_model,'./model_chkpt/lora_adaptor')
    print("Loaded step2 checkpoint")

else:
    projection.load_state_dict(torch.load('./model_chkpt/step1_projection.pth'))
    resblock.load_state_dict(torch.load('./model_chkpt/step1_resblock.pth'))
    print("Loaded step1 checkpoint")


# random validation prediction
def model_run_val(sample_val_data,max_generate_length=10):

    total_val_len = len(sample_val_data)
    random_val_datapoint = random.randrange(1,total_val_len) # 0 is header
    
    val_image_url = sample_val_data[random_val_datapoint][0]
    val_q = sample_val_data[random_val_datapoint][1]
    val_a = sample_val_data[random_val_datapoint][2]

    with torch.no_grad():
        image_load = Image.open(requests.get(val_image_url,stream=True).raw)
        image_processed = processor(images=image_load, return_tensors="pt").to(device)
        clip_val_outputs = clip_model(**image_processed).last_hidden_state[:,1:,:]
        val_image_embeds = projection(clip_val_outputs)
        val_image_embeds = resblock(val_image_embeds).to(torch.float16)
        
        
        img_token_tensor = torch.tensor(IMAGE_TOKEN_ID).to(device)
        img_token_embeds = peft_model.model.model.embed_tokens(img_token_tensor).unsqueeze(0).unsqueeze(0)
        
        val_q_tokenised = tokenizer(val_q, return_tensors="pt", return_attention_mask=False)['input_ids'].squeeze(0)
        val_q_embeds  = peft_model.model.model.embed_tokens(val_q_tokenised).unsqueeze(0)
        
        val_combined_embeds = torch.cat([val_image_embeds, img_token_embeds, val_q_embeds], dim=1) # 1, 69, 2560

        predicted_caption = peft_model.generate(inputs_embeds=val_combined_embeds,
                                                  max_new_tokens=max_generate_length,
                                                  return_dict_in_generate = True)
    
        predicted_captions_decoded = tokenizer.batch_decode(predicted_caption.sequences[:, 1:])[0] 
        predicted_captions_decoded = predicted_captions_decoded.replace("<|endoftext|>", "")

    print(f"Image: {val_image_url}")
    print(f"Question: {val_q}")
    print(f"Answer:   {val_a}")
    print(f"Model Predicted Ans: {predicted_captions_decoded}")
    
model_run_val(sample_val_data,max_generate_length=100)


phi_optimizer        = torch.optim.Adam(peft_model.parameters(), lr=1e-6)
projection_optimizer = torch.optim.Adam(projection.parameters(), lr=1e-5)
resnet_optimizer     = torch.optim.Adam(resblock.parameters(),   lr=1e-5)

step = 0
running_loss = 0.
projection.train()
peft_model.train()
resblock.train()

for epoch in range(1000000):
    for batch_idx, (images,questions,answers) in enumerate(train_dataloader):

        # process input data
        batch_size = questions.size(0)
        questions  = questions.to(device)
        answers    = answers.to(device)

        # clip
        images = {'pixel_values': images.to(device)}
        clip_outputs  = clip_model(**images)
        images_embeds = clip_outputs.last_hidden_state[:,1:,:] # remove cls token
        
        # projection
        image_embeds  = projection(images_embeds)
        image_embeds  = resblock(image_embeds).to(torch.float16)

        # embeds
        #print(questions.shape,answers.shape)
        img_token_tensor = torch.tensor(IMAGE_TOKEN_ID).repeat(batch_size, 1).to(device)
        img_token_embeds = peft_model.model.model.embed_tokens(img_token_tensor)
        questions_embed  = peft_model.model.model.embed_tokens(questions)

        # forward pass
        #print("***************")
        combined_embeds = torch.cat([image_embeds, img_token_embeds, questions_embed], dim=1) # 4, 69, 2560
        #print(f"combined_embeds shape{combined_embeds.shape}")
        phi_output_logits = peft_model(inputs_embeds=combined_embeds)['logits'] # 4, 69, 51200

        # take out the image embeddings
        phi_output_logits = phi_output_logits[:,images_embeds.shape[1] + 1 : ,:]
        #print(f"phi_output_logits after shape{phi_output_logits.shape}")

        phi_optimizer.zero_grad()
        projection_optimizer.zero_grad()
        resnet_optimizer.zero_grad()
        
        loss = F.cross_entropy(phi_output_logits, answers.contiguous().view(-1), ignore_index=50296,label_smoothing=0.1)

        # loss backprop
        loss.backward()
        phi_optimizer.step()
        projection_optimizer.step()
        resnet_optimizer.step()
        

        if step % model_log_step == 0:
            print(f"Iteration {step}/{max_steps}, Loss: {loss.item()}")

        if step % model_val_step == 0:
            projection.eval()
            peft_model.eval()
            resblock.eval()
            model_run_val(sample_val_data,max_generate_length)
            projection.train()
            peft_model.train()
            resblock.train()

        if step % model_save_step == 0:
            print("Saving Checkpoint")
            torch.save(projection.state_dict(),'./model_chkpt/step2_projection.pth')
            torch.save(resblock.state_dict(),'./model_chkpt/step2_resblock.pth')
            peft_model.save_pretrained('./model_chkpt/lora_adaptor/', save_adapter=True, save_config=True)
            
        if step >= max_steps:
            print("Training finished.")
            break
            
        print({"step": step, "train_loss": loss.item()})
        step += 1

