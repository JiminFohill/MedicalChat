import torch
from PIL import Image
import requests

from torch.utils.data import DataLoader
import torch
from finetune_param import TrainAbleParam
from dataloaders.data_helper import QALoader

from transformers import AutoProcessor, AutoTokenizer, Blip2Model

from accelerate import notebook_launcher
from accelerate import Accelerator
import pandas as pd
from tqdm import tqdm
import os

import logging, sys,random
import torch.backends.cudnn as cudnn
import numpy as np

seed=12345
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # z

def save_checkpoint(model,save_path, change_params,iter_num):
    save_param = {}
    all_param = model.state_dict()
    for name_param in all_param:
        if name_param in change_params:
            # print(name_param)
            save_param.update({name_param: all_param[name_param]})
                        
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    finall_save_path = os.path.join(save_path,'query_token_{}.pth'.format(iter_num))
    torch.save(save_param, finall_save_path)
    print('model save')
    
def train():
    
    is_breakpoint = False
    breakpoint_num = 0
    
    model_path = '/hy-tmp/blip2_opt_2.7b'
    train_dir = '/MedicalChat/datasets/raw_and_generate_dataset/train_data.csv'
    save_name = 'query_lm_fc'
    save_path = f'./output/{save_name}'
    logging_path = f'/MedicalChat/logs/{save_name}/{save_name}.txt'
    image_path = ''
    
    max_epoch = 100
    lr = 1e-5
    
    logging.basicConfig(filename=logging_path, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    model = Blip2Model.from_pretrained(model_path).cuda()
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # ------------DataLoader--------------
    dl_train = QALoader(train_dir, image_path)
    
    # ------------model for gradient_checkpointing--------------
    try:
        model.supports_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    except:
        print('Not support gradient_checkpointing')
        
    # -----------Training param--------------
    tap = TrainAbleParam() # tap train able param
    change_params = tap.set_trainable_weights(model)
    print('Training param:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    gradient_accumulation_steps_num = 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps_num)
    iter_num = 0
    
    # 是否断点继训
    if is_breakpoint:
        model.load_state_dict(torch.load(os.path.join(save_path,'query_token_{}.pth'.format(breakpoint_num))), strict=False)
    
    model, optimizer= accelerator.prepare(model, optimizer)
    
    # -----------Training--------------
    epoch_steps = len(dl_train)
    print('total sample:', epoch_steps)

    model.train()
    num = len(dl_train)
    loss_ = 0
    templet = "Question:{} Answer:"
    for epoch_num in tqdm(range(max_epoch)):
        for i in range(num):
            with accelerator.accumulate(model):
                batch = dl_train.next_sample(i)
                image = batch['image']
                question = templet.format(batch['Q'])
                answer = batch['A']

                inputs = processor(images=image, text=question, return_tensors="pt").to(device="cuda", dtype=torch.float32)

                input_ids,labels = dl_train.covert_to_ids(tokenizer, question, answer)
                inputs = {'pixel_values': inputs['pixel_values'], 'input_ids': input_ids.to("cuda"), 'labels': labels.to("cuda")}

                output = model(**inputs)
                loss = output.loss
                #accelerator.backward(loss)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            #lr_scheduler.step()

            iter_num += 1
            loss_+=loss.item()
            if iter_num % gradient_accumulation_steps_num ==0:
                logging.info('iter:%d, loss:%f, epoch:%d' % (iter_num, loss_/gradient_accumulation_steps_num, epoch_num))
                loss_=0

            if iter_num % 6000 == 0:
                save_checkpoint(model,save_path, change_params,iter_num)

        print('epoch:',epoch_num+1)
        save_checkpoint(model,save_path, change_params,iter_num)

notebook_launcher(train, num_processes=1)
# train()