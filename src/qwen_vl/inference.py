import json
import sys
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from src.qwen_vl.inference_data import load_data,InferenceDataset
import torch
import argparse
import random
import logging
logger = logging.getLogger()
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from torch.utils.data import DataLoader, SequentialSampler
from qwen_vl.get_prompt import get_prompt

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__=='__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model_path",type=str,default='Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument('--data_path',type=str,default='')
    parser.add_argument('--retrieval_data_path',type=str,default='')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument("--out_path", type=str, default='')
    parser.add_argument('--topk',type=int,default=0)
    parser.add_argument('--retriever_name',type=str,default='Visualbeg')
    parser.add_argument('--task',type=str,default='') ## {'image_cap', 'mmqa','fact_verify'}
    parser.add_argument('--dataset_name',type=str,default='')

    parser.add_argument('--batch_size',type=int,default=2)
    parser.add_argument('--num_workers',type=int,default=1)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--max_new_tokens',type=int,default=256)
    parser.add_argument('--flag',type=str)


    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_name =args.model_path.split('/')[-1]
    args.out_path=os.path.join(args.out_path,args.dataset_name)
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    set_seed(args)
    tb_writer = SummaryWriter(log_dir=args.out_path)


    args.prompt=get_prompt(args.task,args.topk)

    logger.info(args)
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left",max_pixels =512*512)



    query_data=load_data(args.data_path)
    gen_data=InferenceDataset(args,query_data,args.prompt,processor)
    gen_sampler = SequentialSampler(gen_data)
    gendata_reader = DataLoader(dataset=gen_data, sampler=gen_sampler, num_workers=args.num_workers,
                                  batch_size=args.batch_size, collate_fn=gen_data.Collector, drop_last=False)
    
    out_file=os.path.join(args.out_path,r'{}_{}_{}_results_{}_{}.jsonl'.format(args.model_name,args.task,args.retriever_name,args.topk,args.max_new_tokens))
    if args.flag=='sft':
        out_file=os.path.join(args.out_path,r'{}_{}_{}_{}_results_{}_{}.jsonl'.format(args.model_name,args.task,args.flag,args.retriever_name,args.topk,args.max_new_tokens))
    with open(out_file, 'w') as fin:
        for step, batch in tqdm(enumerate(gendata_reader)):
            with torch.no_grad():
                inputs = batch['inputs']
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                batch_qids = batch['qids']
                for qid, text in zip(batch_qids,output_texts):
                    data = {"id": qid, 'text': text}
                    fin.write(json.dumps(data) + '\n')
                    