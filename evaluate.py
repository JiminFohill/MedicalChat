from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
smooth = SmoothingFunction()
from rouge_score import rouge_scorer
import torch
import numpy as np
from dataloaders.data_helper import QALoader
from dataloaders.data_helper import QALoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import logging
import json
import os
import sys

# generated_text:模型输出
# reference_answer：参考答案
def evaluate_model(generated_text,reference_answer):    
    # BLEU Score
    bleu_scores = []
    smoothing_function = SmoothingFunction().method1
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    for w in weights:
        bleu_score = sentence_bleu([reference_answer.split()], generated_text.split(), weights=w, smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)

    # Rouge Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(generated_text, reference_answer)
        
    return bleu_scores, rouge_scores

def main():
    checkpoint_num = 1212000
    model_path = './blip2_opt_2.7b'
    test_dir = './datasets/raw_and_generate_dataset/test_data.csv'
    save_name = 'query_lm_fc'
    save_path = f'./output/{save_name}'
    checkpoint_path = os.path.join(save_path, 'query_token_{}.pth'.format(checkpoint_num))
    templet = "Question:{} Answer:"
    file_path = f'./output/{save_name}/{save_name}_evaluation_results.json'
    logging_path = f'./output/{save_name}/{save_name}_QA.txt'
    
    dl_test = QALoader(test_dir)
    num = len(dl_test)
    evaluation_results = {}
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_L_scores = []
    
    logging.basicConfig(filename=logging_path, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    model = Blip2ForConditionalGeneration.from_pretrained(model_path).cuda()        
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    processor = AutoProcessor.from_pretrained(model_path)
        
    for i in range(num):
        batch = dl_test.next_sample(i)
        image = batch['image']
        question = templet.format(batch['Q'])
        answer = batch['A']            
        inputs = processor(images=image, text=question, return_tensors="pt").to(device="cuda", dtype=torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        bleu_scores, rouge_scores = evaluate_model(generated_text,answer)
        
        print(f"Q:{question}")
        print(f"A:{generated_text}")
        print(f"R:{answer}")
        
        logging.info(f"Q: {question}")
        logging.info(f"Q: {generated_text}")
        logging.info(f"Q: {answer}")
        
        bleu_1_scores.append(bleu_scores[0])
        bleu_2_scores.append(bleu_scores[1])
        bleu_3_scores.append(bleu_scores[2])
        bleu_4_scores.append(bleu_scores[3])
        rouge_1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge_2_scores.append(rouge_scores['rouge2'].fmeasure)
        rouge_L_scores.append(rouge_scores['rougeL'].fmeasure)
      
    bleu_1_scores_mean = np.mean(bleu_1_scores)
    bleu_2_scores_mean = np.mean(bleu_2_scores)
    bleu_3_scores_mean = np.mean(bleu_3_scores)
    bleu_4_scores_mean = np.mean(bleu_4_scores)            
    rouge_1_scores_mean = np.mean(rouge_1_scores)
    rouge_2_scores_mean = np.mean(rouge_2_scores)
    rouge_L_scores_mean = np.mean(rouge_L_scores)
            
    # 在每次循环结束时计算指标，并将它们添加到字典中
    evaluation_results[f'{save_name}_{checkpoint_num}'] = {
        "BLEU-1": bleu_1_scores_mean,
        "BLEU-2": bleu_2_scores_mean,
        "BLEU-3": bleu_3_scores_mean,
        "BLEU-4": bleu_4_scores_mean,
        "ROUGE-1": rouge_1_scores_mean,
        "ROUGE-2": rouge_2_scores_mean,
        "ROUGE-L": rouge_L_scores_mean
    }
    
    # 打印指标
    # logging.info(f"{breakpoint_num}-BLEU-1 Score: {bleu_1_scores_mean}")
    # logging.info(f"{breakpoint_num}-BLEU-2 Score: {bleu_2_scores_mean}")
    # logging.info(f"{breakpoint_num}-BLEU-3 Score: {bleu_3_scores_mean}")
    # logging.info(f"{breakpoint_num}-BLEU-4 Score: {bleu_4_scores_mean}")
    # logging.info(f"{breakpoint_num}-ROUGE-1 Score: {rouge_1_scores_mean}")
    # logging.info(f"{breakpoint_num}-ROUGE-2 Score: {rouge_2_scores_mean}")
    # logging.info(f"{breakpoint_num}-ROUGE-L Score: {rouge_L_scores_mean}")               

    # 检查文件是否为空
    try:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    except json.decoder.JSONDecodeError:
        existing_data = None
    
    # 判断是否有数据
    if existing_data:
        # 将新的评估结果添加到现有内容中
        existing_data.update(evaluation_results)

        # 将评估结果写入JSON文件
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
    else:
        # 将评估结果写入JSON文件
        with open(file_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
    

if __name__ == "__main__":
    main()
