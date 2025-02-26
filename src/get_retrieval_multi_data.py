import pandas as pd
import random
random.seed(2025)
import base64
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from visual import TSVFile
import json
import logging
import sys
import base64
import os
import io
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
import shutil
import concurrent.futures


def read_image(img_feat_path,img_linelist_path):
    img_map = {}
    img_tsv = []
    all_img_num = 0
    with open(img_linelist_path) as fin:
        for line in fin:
            tokens = line.strip().split('\t')
            all_img_num += 1
            img_map[tokens[0]] = int(tokens[1])
    img_tsv = TSVFile(img_feat_path, all_img_num)
    return img_map,img_tsv

def get_image(img_map,img_tsv,idx):
    offset = img_map[str(idx)]
    img = img_tsv[offset][1]
    return img

def save_images(path,image):
    img_data = base64.b64decode(image)
    image = Image.open(io.BytesIO(img_data)).convert('RGB')
    image.save(path)
    # try:
    #     img_data = base64.b64decode(image)
    #     image = Image.open(io.BytesIO(img_data)).convert('RGB')
    #     image.save(path)
    # except Exception as e:
    #     print(f"Failed to save the image: {e}")
    #

def copy_image_to_target(image_path, target_dir):
    """
        Copies the specified file to the target directory.
        Args:
        image_path (str): The original image file path.
        target_dir (str): The target directory.
    """
    try:

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        file_name = os.path.basename(image_path)
        target_path = os.path.join(target_dir, file_name)
        shutil.copy(image_path, target_path)

    except Exception as e:
        print(f"Error occurred while copying the file: {e}")
    return target_path
    
def read_retrieval_data(retrieval_path):
    qid_2_candid={}
    with open(retrieval_path) as fin:
        for line in fin:
            qid,cand_id,_=line.strip().split('\t')
            if qid not in qid_2_candid:
                qid_2_candid[qid]=[cand_id]
            else:
                qid_2_candid[qid].append(cand_id)
    return qid_2_candid

def load_cand(path,is_image):
    data={}
    with open(path) as fin:
        for line in fin:
            item=json.loads(line.strip())
            id=item['id'] if 'id' in item else item['image_id']
            if is_image: # only used for factify dataset
                image_path=item['image_path']
                data[id]=image_path
            else:
                text=item['text'] if 'text' in item else item['caption']
                data[id]=text
    return data
            
            
def get_cand_data(args,path,task,qid_2_candid,output_path,retrieval_modality=None,cand_multi_path=None,cand_second_path=None):
    retrieval_data=[]
    if retrieval_modality=='multi' and (cand_second_path or cand_multi_path): # for task expect image_rerank
        if 'webqa' in args.dataset_name and cand_second_path: # for mmqa task 
            image_index=0
            img_map,img_tsv=read_image(args.img_feat_path,args.img_linelist_path)
            image_captions=load_cand(cand_second_path,False)
            cand_text_data=load_cand(path,False) 
            for qid,cands in qid_2_candid.items():
                item = {'id':qid, 'cands':[]}
                for did in cands[:args.topN]:
                    if 'text' not in did and 'caption' not in did:
                        image=get_image(img_map,img_tsv,did)
                        output_dir=os.path.join(output_path,f'{args.dataset_name}_{task}_{args.flag}_{retrieval_modality}_retrieval_images_{args.topN}')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        output_name=os.path.join(output_dir,f'image_{image_index}.png')
                        image_index+=1
                        save_images(output_name,image)
                        caption=image_captions[int(did)]
                        item['cands'].append({'image_path':output_name,'image_caption':caption})
                    else:
                        item['cands'].append(cand_text_data[str(did)])
                retrieval_data.append(item)
        elif 'webqa' in args.dataset_name and cand_multi_path: # for image_cap task
            image_index=0
            img_map,img_tsv=read_image(args.img_feat_path,args.img_linelist_path)
            image_captions=load_cand(cand_multi_path,False)
            for qid,cands in qid_2_candid.items():
                item = {'id':qid, 'cands':[]}
                for did in cands[:args.topN]:
                    if 'text' not in did and 'caption' not in did:
                        image=get_image(img_map,img_tsv,did)
                        output_dir=os.path.join(output_path,f'{args.dataset_name}_{task}_{args.flag}_{retrieval_modality}_retrieval_images_{args.topN}')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        output_name=os.path.join(output_dir,f'image_{image_index}.png')
                        image_index+=1
                        save_images(output_name,image)
                        caption=image_captions[int(did)]
                        item['cands'].append({'image_path':output_name,'image_caption':caption})
        elif 'factify' in args.dataset_name and cand_second_path: # for fact_verify task
            cand_text_data=load_cand(path,False)
            cand_image_data=load_cand(cand_second_path,True)
            output_dir=os.path.join(output_path,f'{args.dataset_name}_{task}_{args.flag}_{retrieval_modality}_retrieval_images_{args.topN}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_index=0
            for qid,cands in qid_2_candid.items():
                item = {'id':qid, 'cands':[]}
                for did in cands[:args.topN]:
                    if 'image' in did:
                        image_path=cand_image_data[did]
                        image_path=os.path.abspath(os.path.join(args.data_dir,image_path)) #absolute path
                        new_image_path=copy_image_to_target(image_path, output_dir)
                        item['cands'].append({'image_path':new_image_path})
                    else:
                        item['cands'].append(cand_text_data[str(did)])
                retrieval_data.append(item)
    
    elif retrieval_modality=='image': # used for image rerank task
        img_map,img_tsv=read_image(args.img_feat_path,args.img_linelist_path)
        output_dir=os.path.join(output_path,f'{args.dataset_name}_{task}_{args.flag}_retrieval_images_{args.topN}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_index = 0
        for qid,dids in qid_2_candid.items():
            item = {'id':qid, 'cands':[]}
            for did in dids[:args.topN]:
                image=get_image(img_map,img_tsv,did)
                
                output_name=os.path.join(output_dir,f'image_{image_index}.png')
                image_index+=1
                save_images(output_name,image)
                item['cands'].append({'image_path':output_name})
                if args.topN==1:
                    item = {'id':qid, 'image_path':output_name}
            retrieval_data.append(item)
    output_name=os.path.join(output_path,f'{args.dataset_name}_{task}_{args.flag}_retrieval_{args.retrieval_modality}_{args.topN}.jsonl')
    with open(output_name, "w") as jsonl_file:
        for item in retrieval_data:
            jsonl_file.write(json.dumps(item) + "\n")
    print(f'The retrieval data is writed in {output_name}!')
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_dir",type=str,default='data/m2rag')
    parser.add_argument("--cand_path",type=str,default='')
    parser.add_argument("--cand_second_path",type=str) # the path of the second modality data (mainly used for image)
    parser.add_argument("--cand_multi_path",type=str)
    parser.add_argument("--retrieval_path",type=str,default='')
    parser.add_argument("--img_linelist_path", type=str,default='../data/raw_data/webqa/imgs.lineidx.new')
    parser.add_argument("--img_feat_path", type=str, default='../data/raw_data/webqa/imgs.tsv')
    
    parser.add_argument("--dataset_name", type=str, default='webqa')
    parser.add_argument("--output_path", default='../output/retrieval/')
    parser.add_argument('--retriever_name', type=str,default='visualbge')
    parser.add_argument('--task',type=str,default='image_cap') #{'image_cap', 'mmqa', 'fact_verify', 'image_rerank'}
    parser.add_argument('--flag', type=str, default='test')
    parser.add_argument('--topN', type=int, default=1)
    parser.add_argument('--retrieval_modality', type=str, default='multi') #{'text','image','multi'}
    



    args = parser.parse_args()

    args.output_dir=os.path.join(args.output_path,args.task)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    qid_2_candid=read_retrieval_data(args.retrieval_path)
    print("-------------------")
    get_cand_data(args,args.cand_path,args.task,qid_2_candid,args.output_dir,args.retrieval_modality,args.cand_multi_path,args.cand_second_path)
            
        
                
                





            