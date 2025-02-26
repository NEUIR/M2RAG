import pandas as pd
import json
import random
random.seed(2025)
import os
import base64
from PIL import Image
from src.visual import TSVFile
import io
import json


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
    print(f"Image saved in {path}")
        

def read_score_data(path,k=0.65):
    indexs=[]
    with open(path, 'r') as fin:
        for line in fin:
            example=json.loads(line.strip())
            if example['score'] >=k:
                indexs.append(example['id'])
            if len(indexs)==6000:
                break
    return indexs

def get_cand_data(path,indexs,output_path):
    cand_data=[]
    captions_data=[]
    with open(path,'r') as fin:
        for index,line in enumerate(fin):
            if index not in indexs:
                example=json.loads(line.strip())
                cand_data.append(example)
                caption=example['caption']
                captions_data.append({'id':f'caption_{index}','text':caption})
    image_path=os.path.join(output_path,'all_imgs_new.json')
    write_data(cand_data,image_path)
    # captions_path=os.path.join(output_path,'cand_captions.json')
    # write_data(captions_data,captions_path)
    
    

def get_data(path,indexs,output_path,img_map,img_tsv):
    train_data = []
    test_data = []
    id=0
    with open(path, 'r') as fin:
        for index,line in enumerate(fin):
            if index not in indexs:
                continue
            if index>indexs[-1]:
                break
            example=json.loads(line.strip())
            caption=example['caption']
            image_id=example['image_id']
            image=get_image(img_map,img_tsv,image_id)
            if id<3000:
                item_id='train_'+str(id)
                image_path=os.path.join(output_path,'train_images/train_img_{}.png'.format(id))
            else:
                item_id='test_'+str(id-3000)
                image_path=os.path.join(output_path,'test_images/test_img_{}.png'.format(id-3000))
            save_images(image_path,image)
            item={'id':item_id,'caption':caption,'image_path':image_path}
            if id<3000:
                train_data.append(item)
            else:
                test_data.append(item)
            id+=1
    assert len(train_data)==3000 and len(test_data)==3000
    return train_data, test_data

def write_data(data,outpath):
    with open(outpath, 'w') as fout:
        for item in data:
            json.dump(item, fout) 
            fout.write('\n')
    print(f"Data has been written to {outpath}")
 

if __name__=='__main__':
    data_path='../raw_data/WebQA/all_imgs.json'
    img_feat_path='../raw_data/WebQA/imgs.tsv'
    img_linelist_path='../raw_data/WebQA/imgs.lineidx.new'
    score_path='../raw_data/WebQA/all_image_caption_score.jsonl'
    output_path='../m2rag/image_cap'
    os.makedirs(output_path, exist_ok=True)
    
    indexs=read_score_data(score_path)
    assert len(indexs)==6000
    img_map,img_tsv=read_image(img_feat_path,img_linelist_path)
    
    train_data, test_data=get_data(data_path,indexs,output_path,img_map,img_tsv)
    cand_image_captions=get_cand_data(data_path,indexs,output_path)
    data_output_path1=os.path.join(output_path,'train_data.jsonl')
    data_output_path2=os.path.join(output_path,'test_data.jsonl')
    write_data(train_data,data_output_path1)
    write_data(test_data,data_output_path2)

