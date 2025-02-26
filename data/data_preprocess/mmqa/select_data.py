import sys
import pandas as pd
import json
import random
random.seed(2025)
import os
import base64
from PIL import Image
from src.visual import TSVFile
import io
import shutil



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


def load_doc(path):
    data={}
    with open(path,'r') as fin:
        for line in fin:
            item=json.loads(line)
            if 'snippet_id' in item:
                data[item['snippet_id']]=item['fact']
            elif 'image_id' in item:
                data[str(item['image_id'])]=item['caption'] #item['image_id'] is int
        return data
            
def save_images(path,image):
    img_data = base64.b64decode(image)
    image = Image.open(io.BytesIO(img_data)).convert('RGB')
    image.save(path)
    print(f"Image saved in {path}")
    
def load_data(path,output_path,img_map,img_tsv,text_doc,captions,is_train=False):
    id=0
    tmpid=0
    # the following para is used to record the num of golden doc
    image_num=0
    text_num=0
    data_list=[]
    data_load=[]
    with open(path) as fin:
        for index,line in enumerate(fin):
            data=json.loads(line.strip())
            data_load.append(data)
    random.shuffle(data_load)
    if is_train:
        image_dir=os.path.join(output_path,'train_images')
    else:
        image_dir=os.path.join(output_path,'test_images')
    os.makedirs(image_dir, exist_ok=True)
    for data in data_load:
        if len(data_list)>=3000:
            break
        if data['img_posFacts']!=[] and image_num<1500:
            image_id = random.choice(data['img_posFacts'])
            image_path=os.path.join(image_dir,f'image_{tmpid}.png')
            tmpid+=1
            image=get_image(img_map,img_tsv,image_id)
            save_images(image_path,image)
            data_list.append({'id': f'test_{id}', 'pos_image_path': image_path, 'pos_image_caption': captions[image_id], 'query': data['Q'],'answer': data['A']})
            id+=1
            image_num+=1
        if data['txt_posFacts']!=[] and text_num<1500:
            text_doc_id=random.choice(data['txt_posFacts'])
            data_list.append({'id': f'test_{id}', 'pos_text': text_doc[text_doc_id], 'query': data['Q'],'answer': data['A']})
            id+=1
            text_num+=1
                             
    return data_list

def write_data(data,path,is_train=False):
    if is_train:
        path=os.path.join(path,'train_data.jsonl')
    else:
        path=os.path.join(path,'test_data.jsonl')
    with open(path, mode='w', encoding='utf-8') as fout:
        for item in data:
            json.dump(item, fout, ensure_ascii=False)
            fout.write('\n')
    print(f'the data is written in {path}')
    
    
def write_doc_data(text_doc_path,path):
    data=[]
    with open(text_doc_path,'r') as fin:
        for index, line in enumerate(fin):
            item=json.loads(line)
            example={"id": f"text_{index}", "text":item["fact"]}
            data.append(example)
            
    path=os.path.join(path,'cand_text_docs.jsonl')
    with open(path, mode='w', encoding='utf-8') as fout:
        for item in data:
            json.dump(item, fout)
            fout.write('\n')
    print(f'the data is written in {path}')

if __name__=='__main__':
    test_data_path='../raw_data/WebQA/test.json'
    train_data_path='../raw_data/WebQA/train.json'
    img_feat_path='../raw_data/WebQA/imgs.tsv'
    img_linelist_path='../raw_data/WebQA/imgs.lineidx.new'
    text_doc_path='../raw_data/WebQA/all_docs.json'
    image_cap_path='../raw_data/WebQA/all_imgs.json'
    output_path='../m2rag/mmqa'
    os.makedirs(output_path, exist_ok=True)
    
    img_map,img_tsv=read_image(img_feat_path,img_linelist_path)
    text_doc=load_doc(text_doc_path)
    captions=load_doc(image_cap_path)
    
    #test
    data_list=load_data(test_data_path,output_path,img_map,img_tsv,text_doc,captions)
    write_data(data_list,output_path)
    
    # train
    data_list=load_data(train_data_path,output_path,img_map,img_tsv,text_doc,captions,True)
    write_data(data_list,output_path,True)
    
    # cand_text_doc
    write_doc_data(text_doc_path,output_path) 
    
    # cand_image_path
    shutil.copy(image_cap_path, output_path)
