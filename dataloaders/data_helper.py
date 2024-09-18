from torch.utils.data import Dataset
import json
from tqdm import tqdm
import transformers
import random
import torch
import pandas as pd
from PIL import Image

class QALoader:

    def __init__(self, base_dir, image_path):
        self.longest =200

        df = pd.read_csv(base_dir)
        self.image_path = image_path
        self.study_id = df['study_id']
        self.subject_id = df['subject_id']
        self.dicom_id = df['dicom_id']
        self.question = df['question']
        self.answer = df['answer']

    def __len__(self):

        return len(self.question)

    def next_sample(self, idx):

        sid = self.study_id[idx]
        subid = self.subject_id[idx]
        dicomid = self.dicom_id[idx]

        Q = self.question[idx]
        A = self.answer[idx]

        image_path = f"{image_path}\\p{str(subid)[:2]}\\p{subid}\\s{sid}\\{dicomid}.jpg"

        image = Image.open(image_path)
        sample = {'image':image, 'Q':Q, 'A':A}
        return sample

    def covert_to_ids(self,tokenizer, context, target, max_seq_length=100):

        context_ids = tokenizer.encode(
            context,
            max_length=max_seq_length,
            truncation=True)

        target_ids = tokenizer.encode(
            target,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False)

        ids = context_ids + target_ids + [1] # pad id

        context_len = len(context_ids)

        labels = (
                [-100] * (context_len - 1) + ids[(context_len - 1):] + [-100] * (self.longest - len(ids))
        )  # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss

        ids = ids + [1] * (self.longest - len(ids))

        input_ids = torch.LongTensor([ids])
        labels = torch.LongTensor([labels])

        return  input_ids,labels
    
class VQA_RAD_Loader:

    def __init__(self, base_dir):
        self.longest =200

        df = pd.read_csv(base_dir)

        self.image_name = df['image_name']
        self.image_organ = df['image_organ']
        self.question = df['question']
        self.question_type = df['question_type']
        self.answer = df['answer']
        self.answer_type = df['answer_type']

    def __len__(self):

        return len(self.question)

    def next_sample(self, idx):

        
        image_name = self.image_name
        Q = self.question[idx]
        A = self.answer[idx]
        question_type = self.question_type[idx]
        answer_type = self.answer_type[idx]
        image_path = "D:\\MedicalChat\\datasets\\VQA-RAD\\osfstorage-archive\\VQA_RAD_Image_Folder\\"+image_name

        image = Image.open(image_path)
        sample = {'image':image, 'Q':Q, 'A':A, 'question_type':question_type, 'answer_type':answer_type}
        return sample

    def covert_to_ids(self,tokenizer, context, target, max_seq_length=100):

        context_ids = tokenizer.encode(
            context,
            max_length=max_seq_length,
            truncation=True)

        target_ids = tokenizer.encode(
            target,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False)

        ids = context_ids + target_ids + [1] # pad id

        context_len = len(context_ids)

        labels = (
                [-100] * (context_len - 1) + ids[(context_len - 1):] + [-100] * (self.longest - len(ids))
        )  # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss

        ids = ids + [1] * (self.longest - len(ids))

        input_ids = torch.LongTensor([ids])
        labels = torch.LongTensor([labels])

        return  input_ids,labels
