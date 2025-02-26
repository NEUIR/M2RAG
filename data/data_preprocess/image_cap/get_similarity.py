"""
Step-1: Preprocess the data to obtain similarity scores between images and captions, and filter out pairs with higher similarity scores.
"""
import sys
from src.visual_bge.modeling import Visualized_BGE
from visual import TSVFile
import json
import logging
import sys
import base64
import os
import pandas as pd
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import clip
logger = logging.getLogger()
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WebqaDataset(Dataset):
    def __init__(self, args, preprocess,tokenizer,data=None,indexs=None,is_clip=False):

        self.data = data #caption
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.indexs=indexs
        self.is_clip=is_clip

        self.img_map = {}
        self.img_tsv = []
        img_feat_path = args.img_feat_path
        img_linelist_path = args.img_linelist_path
        all_img_num = 0
        with open(img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
        self.img_tsv = TSVFile(img_feat_path, all_img_num)

    def __len__(self):
        return len(self.data)

    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = Image.open(io.BytesIO(base64.b64decode(img))).convert('RGB')
        img = self.preprocess(img)
        return {'img': img}


    def Collector(self, batch):
        img_inputs = []
        captions=[]
        idx_list = []

        for example in batch:
            img_inputs.append(example['img_inputs'])
            captions.append(example['caption'])
            idx_list.append(example['idx'])
        processed_batch = {}
        processed_batch['idx_list'] = idx_list
        processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)
        if self.is_clip:
            processed_batch['captions_inputs']= self.tokenizer(captions, truncate=True)
        else:
            processed_batch['captions_inputs']= self.tokenizer(captions,return_tensors="pt", padding=True)

        return processed_batch

    def __getitem__(self,index):
        item = self.data[index]
        img_input = self.encode_img(str(item['image_id']))
        instance = {
            'idx': index,
            'img_inputs': img_input['img'],
            'caption' : item['caption'],
        }

        return instance


def get_similarity_score(model, valid_reader, outpath):
    model.eval()
    all_scores = []
    all_index = []
    for step, batch in tqdm(enumerate(valid_reader)):
        with torch.no_grad():
            idx_list = batch['idx_list']
            image_embeddings = model.encode_image(batch['img_inputs'].cuda())
            text_embeddings = model.encode_text(batch['captions_inputs'].to(device))
            # scores1=model.compute_similarity(text_embeddings,image_embeddings)
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
            scores=model.compute_similarity(text_embeddings,image_embeddings).diagonal()
            assert len(scores) == len(idx_list)
            all_index.extend(idx_list)
            tmp=scores.squeeze().cpu().tolist()
            all_scores.extend(scores.squeeze().cpu().tolist())

    with open(outpath, 'w') as fout:
        for idx, score in zip(all_index, all_scores):
            data = {'id': idx, 'score': score}
            json.dump(data, fout) 
            fout.write('\n') 
    print('the score is write to {}'.format(outpath))




def load_data(path):
    data = []
    if 'webqa' in path.lower():
        with open(path, 'r') as fin:
            for line in fin:
                data.append(json.loads(line.strip()))
    return data


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")

    parser.add_argument("--out_path", type=str,default='../raw_data/WebQA/')
    parser.add_argument("--model_name", type=str,default="../../pretrained_model/bge-base-en-v1.5")
    parser.add_argument("--model_path",type=str,default="../../pretrained_model/Visualized_base_en_v1.5.pth")
    parser.add_argument("--max_seq_len", type=int)

    parser.add_argument("--cap_path", type=str,default='../raw_data/WebQA/all_imgs.json')
    parser.add_argument("--img_feat_path", type=str,default='../raw_data/WebQA/imgs.tsv')
    parser.add_argument("--img_linelist_path", type=str,default='../raw_data/WebQA/imgs.lineidx.new')

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    if 'clip' in args.model_name:
        model, preprocess = clip.load("ViT-B/32", device=device)  # Must set jit=False for training
        tokenizer=clip.tokenizer
        model.cuda()
    else:
        model = Visualized_BGE(model_name_bge = args.model_name, model_weight=args.model_path).cuda()
        preprocess=model.preprocess_val
        tokenizer=model.tokenizer
    captions = load_data(args.cap_path)

    data = WebqaDataset(args, preprocess, tokenizer, captions)
    data_sampler = SequentialSampler(data)
    data_reader = DataLoader(dataset=data, sampler=data_sampler, num_workers=args.num_workers,
                                batch_size=args.batch_size, collate_fn=data.Collector)

    outpath = os.path.join(args.out_path, 'all_image_caption_score.jsonl')
    get_similarity_score(model, data_reader, outpath)