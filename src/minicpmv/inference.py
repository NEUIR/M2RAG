import json
import sys
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from minicpmv.inference_data import load_data,InferenceDataset
import torch
import argparse
import random
import logging
logger = logging.getLogger()
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from minicpmv.get_prompt import get_prompt
from peft import PeftModel

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__=='__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model_path",type=str,default='openbmb/MiniCPM-V-2_6')
    parser.add_argument("--path_to_adapter", type=str)
    parser.add_argument('--data_path',type=str,default='')
    parser.add_argument('--retrieval_data_path',type=str,default='')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument("--out_path", type=str, default='')
    parser.add_argument('--topk',type=int,default=1)
    parser.add_argument('--retriever_name',type=str,default='Visualbeg')
    parser.add_argument('--task',type=str,default='mmqa') ## {'image_cap', 'mmqa','fact_verify'}
    parser.add_argument('--dataset_name',type=str,default='')

    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--num_workers',type=int,default=1)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--max_new_tokens',type=int,default=64)
    parser.add_argument('--max_slice_nums',type=int,default=9)
    parser.add_argument('--flag',type=str)
    


    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_name =args.model_path.split('/')[-1]
    args.out_path=os.path.join(args.out_path,args.task)
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
    
    logger.info(args)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    if args.path_to_adapter:
        model = PeftModel.from_pretrained(
                                            model,
                                            args.path_to_adapter,
                                            device_map="auto",
                                            trust_remote_code=True
                                        )
        
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)


    query_data=load_data(args.data_path)
    gen_data=InferenceDataset(args,query_data,args.prompt)
    gen_sampler = SequentialSampler(gen_data)
    gendata_reader = DataLoader(dataset=gen_data, sampler=gen_sampler, num_workers=args.num_workers,
                                  batch_size=args.batch_size, collate_fn=gen_data.Collector, drop_last=False)

    out_file=os.path.join(args.out_path,r'{}_{}_{}_{}_results_{}.jsonl'.format(args.model_name,args.dataset_name,args.task,args.retriever_name,args.topk))
    if args.flag=='sft':
        out_file=os.path.join(args.out_path,r'{}_{}_{}_{}_{}_results_{}.jsonl'.format(args.model_name,args.dataset_name,args.task,args.flag, args.retriever_name,args.topk))
    params={}
    if args.max_slice_nums != 9:
        params["max_slice_nums"] = args.max_slice_nums
    
    with open(out_file, 'w') as fin:
        for step, batch in tqdm(enumerate(gendata_reader)):
            with torch.no_grad():
                msgs_list = batch['msgs_list']
                new_contexts = []
                for msgs in msgs_list:
                    answer = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        **params
                    )
                    new_contexts.append(answer)

                batch_qids = batch['qids_list']
                for qid, text in zip(batch_qids,new_contexts):
                    data = {"id": qid, 'text': text}
                    fin.write(json.dumps(data) + '\n')
                    