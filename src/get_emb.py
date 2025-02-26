import json
import logging
import sys
import base64
import os
from typing import Optional
import numpy as np
from torch import nn
from torch.nn import LayerNorm
from tqdm import tqdm
import torch
import argparse
import os.path as op
import time
import pickle
import math
import base64
from PIL import Image
import io
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from src.visual_bge.modeling import Visualized_BGE
from src.visual import TSVFile
logger = logging.getLogger()
import random
from PIL import ImageFile, Image
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImgDataset(Dataset):
    def __init__(self, args, preprocess, tokenizer, max_len,images=None, is_query=None):

        self.images = images
        self.preprocess = preprocess
        self.task= args.task
        self.is_query=is_query
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self):
        return len(self.images)

    def encode_img(self, image_path):
        image_path=os.path.abspath(os.path.join(args.data_dir,image_path)) #absolute path
        img = self.preprocess(Image.open(image_path).convert('RGB'))
        return {'img': img}


    def Collector(self, batch):
        img_inputs = []
        idx_list = []
        cap_inputs = []

        for example in batch:
            img_inputs.append(example['img_inputs'])
            idx_list.append(example['idx'])
            if 'caption' in example:
                cap_inputs.append(example['caption'])
        processed_batch = {}
        processed_batch['idx_list'] = idx_list
        processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)

        if len(cap_inputs)!=0:
            processed_batch['cap_inputs']=self.tokenizer(cap_inputs, return_tensors='pt', max_length=self.max_len,
                                                         padding='max_length', truncation=True)

        return processed_batch

    def __getitem__(self, index):
        item = self.images[index]
        
        if self.task=='fact_verify' and self.is_query: # fact query
            img_inputs = self.encode_img(item['claim_image'])
        else:
            img_inputs = self.encode_img(item['image_path']) # image_cap query and fact image_cands

        instance = {
        'idx': item['id'],
        'img_inputs': img_inputs['img']
        }
        if self.task=='fact_verify' and self.is_query:
            instance['caption']=item['claim']
        return instance

class WebqaImgDataset(Dataset):
    def __init__(self, args, preprocess, tokenizer, max_len, data=None):

        self.img_map = {}
        self.img_ids = []
        self.data = data 
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task= args.task 
        
        all_img_num = 0
        with open(args.img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
                self.img_ids.append(tokens[0])
        self.img_tsv = TSVFile(args.img_feat_path, all_img_num)

    def __len__(self):
        return len(self.data)

    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = self.preprocess(Image.open(io.BytesIO(base64.b64decode(img))).convert('RGB'))
        return {'img': img}


    def Collector(self, batch):
        img_inputs = []
        idx_list = []
        cap_inputs = []

        for example in batch:
            img_inputs.append(example['img_inputs'])
            idx_list.append(example['idx'])
            if 'caption' in example:
                cap_inputs.append(example['caption'])
        processed_batch = {}
        processed_batch['idx_list'] = idx_list
        processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)

        if len(cap_inputs)!=0:
            processed_batch['cap_inputs']=self.tokenizer(cap_inputs, return_tensors='pt', max_length=self.max_len,
                                                         padding='max_length', truncation=True)

        return processed_batch

    def __getitem__(self, index):
        item=self.data[index]
        img_idx = item['image_id']
        
        img_inputs = self.encode_img(str(img_idx))
        instance = {
            'idx': img_idx,
            'img_inputs': img_inputs['img']
        }
        if (self.task=='mmqa' or self.task=='image_cap') and self.data: # image_cap and mmrag cand_images
            instance['caption']=item['caption']
        return instance

   
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len,is_query=False):
        self.max_len = max_len
        self.data = data
        self.tokenizer = tokenizer
        self.is_query = is_query

    def __len__(self):
        return len(self.data)


    def Collector(self, batch):
        txt_inputs = []
        idx_list = []

        for qid, example in enumerate(batch):
            txt_inputs.append(example['txt_inputs'])
            idx_list.append(example['idx'])
        processed_batch = {
            'txt_inputs': self.tokenizer(txt_inputs, return_tensors="pt", max_length=self.max_len, padding='max_length',truncation=True),
            'idx_list': idx_list
        }
        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        if self.is_query: # image reranking or mmrag query
            txt_inputs = example['caption'] if 'caption' in example else example['query']
        else:
            txt_inputs = example['text'] # mmqa or factiry cands
        return {
            'idx': example['id'],
            'txt_inputs': txt_inputs
        }

def gen_embeddings(model, cand_reader, outpath):
    model.eval()
    all_embeddings = []
    all_index = []
    for step, batch in tqdm(enumerate(cand_reader)):
        with torch.no_grad():
            idx_list = batch['idx_list']
            if 'img_inputs' in batch and 'cap_inputs' in batch: # image+cap
                embeddings = model.encode_mm(batch['img_inputs'].to('cuda'),batch['cap_inputs'].to('cuda'))
            elif 'img_inputs' in batch and 'cap_inputs' not in batch: # image
                embeddings = model.encode_image(batch['img_inputs'].to('cuda'))
            else: # text
                embeddings = model.encode_text(batch['txt_inputs'].to('cuda'))
            embeddings = F.normalize(embeddings, dim=-1)
            embeddings = embeddings.cpu()
            assert len(embeddings) == len(idx_list)
            all_index.extend(idx_list)
            all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    with open(outpath, 'wb') as fout:
        pickle.dump((all_index, all_embeddings), fout)

def load_data(path):
    data = []
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            data.append(example)
    return data

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")

    parser.add_argument("--out_path", type=str,default='../output/embed/')
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--model_weight", type=str, default="../pretrained_model/Visualized_base_en_v1.5.pth")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--dataset_name", type=str, default='webqa')
    parser.add_argument("--task", type=str, default='mmqa') #{'image_cap', 'mmqa', 'fact_verify', 'image_rerank'}
    parser.add_argument('--is_query', action='store_true', default=False)
    parser.add_argument('--flag', type=str, default='test')

    parser.add_argument("--data_dir", type=str,default='data/m2rag')
    parser.add_argument("--text_path", type=str,default='')
    parser.add_argument("--image_path", type=str,default='')
    parser.add_argument("--img_linelist_path", type=str,default='../data/m2rag/imgs.lineidx.new')
    parser.add_argument("--img_feat_path", type=str, default='../data/m2rag/imgs.tsv')
    parser.add_argument("--sec_text_path", type=str,default='')

    parser.add_argument('--encode_text', action='store_true', default=False)
    parser.add_argument('--encode_image', action='store_true', default=False)

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    args.output_dir=os.path.join(args.out_path,args.task)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    handlers = [logging.FileHandler(os.path.join(args.output_dir, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    model = Visualized_BGE(model_name_bge = args.model_name , model_weight = args.model_weight)
    model.eval()
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    preprocess=model.preprocess_val
    
    
    if args.encode_text and args.text_path: 
        txt_datas = load_data(args.text_path)
        txt_data = TextDataset(txt_datas, tokenizer, args.max_len,args.is_query)
        txt_sampler = SequentialSampler(txt_data)
        txt_reader = DataLoader(dataset=txt_data, sampler=txt_sampler, num_workers=args.num_workers,
                                    batch_size=args.batch_size, collate_fn=txt_data.Collector)

        if 'webqa' in args.dataset_name and args.task=='mmqa' and not args.is_query: # mmqa doc cands
            output = os.path.join(args.output_dir, f'{args.dataset_name}_cands_text_docs_embedding.pkl')
        elif args.is_query: # image_reranking query or mmqa query
            output = os.path.join(args.output_dir, f'{args.dataset_name}_{args.task}_query_text_{args.flag}_embedding.pkl')
        else: # factify cands
            output = os.path.join(args.output_dir, f'{args.dataset_name}_cands_text_embedding.pkl')
        gen_embeddings(model, txt_reader, output)
        
    if args.encode_image:
        images = load_data(args.image_path)
        if 'webqa' in args.dataset_name and not args.is_query: #image caption (cands: img+cap) or image reranking (cands: img) or MMQA(cands: img+cap)
            images_data=WebqaImgDataset(args, preprocess, tokenizer, args.max_len, images)
        else:
            images_data = ImgDataset(args, preprocess, tokenizer, args.max_len, images, args.is_query) # image caption (query)  or fact verify (query and cand image)
        images_sampler = SequentialSampler(images_data)
        images_reader = DataLoader(dataset=images_data, sampler=images_sampler, num_workers=args.num_workers,
                                    batch_size=args.batch_size, collate_fn=images_data.Collector)
        if (args.task=='mmqa' or args.task=="image_cap") and not args.is_query: # MMQA cands and image_cap cands
            output = os.path.join(args.output_dir, f'{args.dataset_name}_cands_images_caps_embedding.pkl') 
        elif args.is_query and args.task=='fact_verify': # Fact check query(claim + claim_image)
            output = os.path.join(args.output_dir, f'{args.dataset_name}_{args.task}_query_claim_images_{args.flag}_embedding.pkl')
        elif args.is_query: # image_caption query
            output = os.path.join(args.output_dir, f'{args.dataset_name}_{args.task}_query_images_{args.flag}_embedding.pkl')
        else: #image rerank cands  or fact cands
            output = os.path.join(args.output_dir, f'{args.dataset_name}_cands_images_embedding.pkl')
        gen_embeddings(model, images_reader, output)
    
    
    