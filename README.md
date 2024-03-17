# era_capstone

The following is the link of the final hugging face application
https://huggingface.co/spaces/saurabhmangal12/Multimodal_LLM_ERA

Providing 2 examples of the same:

Query 1
![Image Description](https://github.com/saurabhmangal/era_capstone/blob/main/Query%201.jpg)

Query 2
![Image Description](https://github.com/saurabhmangal/era_capstone/blob/main/Query%202.JPG)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Assignment 1: Training Microsoft Phi2 GPT model from scratch
Description: I have used sample of redpajama dataset where I have used components in the following way:

data_config = [("arxiv_sample_00000000", 33.0),("book_sample_00000000", 33.0),("c4_sample_00000000", 29.0),("generated_data_00000000", 5.0)]

Here the generated data is the data generated from phi2 model. In the assignment it was said to generate the data while training. However, my computation was becoming too slow and subsequently I have generated the data and then used for training. 

The following are the logs for the same:
[Link Text](https://github.com/saurabhmangal/era_capstone/blob/main/1initial_training_phase1/log.txt)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Assignment 2 Step 1: Pretrain for annotated images.

I have used Coco dataset for pretraining CLIP model (clip-vit-base-patch32) and LLM model (Phi2). 
The following are the training logs:
[Link Text](https://github.com/saurabhmangal/era_capstone/blob/main/2model_pretraining/log1.txt)
[Link Text](https://github.com/saurabhmangal/era_capstone/blob/main/2model_pretraining/log2.txt)
[Link Text](https://github.com/saurabhmangal/era_capstone/blob/main/2model_pretraining/log3.txt)
[Link Text](https://github.com/saurabhmangal/era_capstone/blob/main/2model_pretraining/log4.txt)
[Link Text](https://github.com/saurabhmangal/era_capstone/blob/main/2model_pretraining/log5.txt)
[Link Text](https://github.com/saurabhmangal/era_capstone/blob/main/2model_pretraining/log6.txt)

Example Results:
0 - Target captions:
 There is only room for one person to sit in the chair.  
0 - predicted_captions:
 A living room with a couch,<|endoftext|> 
1 - Target captions:
 A batter keeps his arm loose as he waits at home plate.  
1 - predicted_captions:
 A baseball game with a batter, a pitcher<|endoftext|> 

Note: For all these computation I have NVIDIA DGX machine so you will also find proxy bypas in most of the codes. This is a machine in my office environment and I have taken proper permissions to use the same. 
![Image Description](https://github.com/saurabhmangal/era_capstone/blob/main/dgx.JPG)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Assignment 2 Step 2: Finetuning for images and start Q&A mode.

The dataset that I have used is Instruct150k Dataset

The following is the training log:
[Link Text](https://github.com/saurabhmangal/era_capstone/blob/main/3finetuning/log.txt)


Example predictions
Image: http://images.cocodataset.org/train2017/000000051587.jpg
Question: Where is the empty suitcase located? [QA]
Answer:   The empty suitcase is located on the wood floor in front of a radiator.<|endoftext|>
Model Predicted Ans:  empty suitcase is located on the floor, next to a bed.

