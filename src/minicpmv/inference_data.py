import base64
import json
import io
import torch
from torch.utils.data import Dataset
import os
from PIL import ImageFile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_data(path):
    data=[]
    with open(path) as fin:
        for line in fin:
            data.append(json.loads(line.strip()))
    return data

def load_retrieval_context(path):
    query_caption_map={}
    with open(path) as fin:
        for line in fin:
            item=json.loads(line.strip())
            qid=item['id']
            if qid not in query_caption_map:
                query_caption_map[qid]=item['retrieval_text'] if 'retrieval_text' in item else item['cands']
    return query_caption_map

def construct_image_template_for_minicpmv(image_list):
    image_template = ''
    for i,image_path in enumerate(image_list):
        image_template += f'<image_0{i}>\n'
    return image_template

class InferenceDataset(Dataset):
    def __init__(self, args, data, prompt):
        
        self.data=data
        self.prompt=prompt
        self.topk = args.topk
        self.device=args.device
        self.task=args.task  # {'image_cap', 'mmqa','fact_verify'}
        if args.retrieval_data_path!='' and self.topk!=0:
            self.query_caption_map=load_retrieval_context(args.retrieval_data_path)


    def __len__(self):
        return len(self.data)

    def encode_img(self, img):
        raw_image =Image.open(img).convert('RGB')
        return raw_image

    def Collector(self, batch):
        processed_batch = {}
        msgs_list=[]
        qids_list=[]
        for index, example in enumerate(batch):
            qids_list.append(example['id'])
            if self.topk!=0:
                if self.task=='image_cap':
                    content=[example['image']]
                    if 'cands_image' in example:
                        content=content+example['cands_image']
                    image_template=construct_image_template_for_minicpmv(content)
                    if 'cands_caption' in example:
                        retrirval_image_caption=example['cands_caption']
                    elif 'cands_text' in example:
                        retrirval_image_caption=example['cands_text']
                    else:
                        retrirval_image_caption='null'
                    text_prompt=self.prompt.replace("Image: The first image.",f"{image_template}\nImage: The first image.")
                    text_prompt=text_prompt.format(retrirval_image_caption=retrirval_image_caption)
                    
                elif self.task=='mmqa':
                    content=[]
                    retrirval_text=example['cands_text'] if 'cands_text' in example else 'null'
                    if 'cands_image' in example:
                        content=content+example['cands_image']
                    image_template=construct_image_template_for_minicpmv(content)
                    retrirval_image_caption=example['cands_caption'] if 'cands_caption' in example else 'null'
                    
                    text_prompt=self.prompt.replace("Question:",f"{image_template}\nQuestion:")
                    text_prompt=text_prompt.format(question=example['query'], retrirval_text=retrirval_text, retrirval_image_caption=retrirval_image_caption)

                elif self.task=='fact_verify':
                    content=[example['claim_image']]
                    retrirval_text=example['cands_text'] if 'cands_text' in example else 'null'
                    if 'cands_image' in example:
                        content=content+example['cands_image']
                    image_template=construct_image_template_for_minicpmv(content)
                    text_prompt=self.prompt.replace("Claim_Image: The first image.",f"{image_template}\nClaim_Image: The first image.")
                    text_prompt=text_prompt.format(claim_text=example['claim'], retrirval_text=retrirval_text)
                
                else:
                    raise ValueError("The task is error!")
                
            else:
                if self.task=='image_cap':
                    content=[example['image']]
                    image_template=construct_image_template_for_minicpmv(content)
                    
                    text_prompt=self.prompt.replace("Image: The first image.",f"{image_template}\nImage: The first image.")
                
                elif self.task=='mmqa':
                    content=[]
                    image_template=construct_image_template_for_minicpmv(content)
                    
                    text_prompt=self.prompt.replace("Question:",f"{image_template}\nQuestion:")
                    text_prompt=text_prompt.format(question=example['query'])

                elif self.task=='fact_verify':
                    content=[example['claim_image']]
                    image_template=construct_image_template_for_minicpmv(content)
                    
                    text_prompt=self.prompt.replace("Claim_Image: The first image.",f"{image_template}\nClaim_Image: The first image.")
                    text_prompt=text_prompt.format(claim_text=example['claim'])
                
                else:
                    raise ValueError("The task is error!")
                
            content.append(text_prompt)
            msgs = [{"role":'user', "content":content}]
            msgs_list.append(msgs)
        processed_batch['msgs_list']=msgs_list
        processed_batch['qids_list']=qids_list
        return processed_batch


    def __getitem__(self, index):
        example = self.data[index]
        instance = {}
        instance['id'] = example['id']
        if self.task=='image_cap':
            instance['image']=self.encode_img(example['image_path'])
            
        elif self.task=='mmqa':
            instance['query']=example['query']
            
        elif self.task=='fact_verify':
            instance['claim']=example['claim']
            instance['claim_image']=self.encode_img(example['claim_image'])
            
            document=example['document'].split()
            document=document[:500]
            document=' '.join(document)
            instance['document']=document
            
            instance['document_image']=self.encode_img(example['document_image'])

        else:
                    raise ValueError("The task is error!")

        if self.topk!=0:
            cands=self.query_caption_map[str(example['id'])]
            
            cands_text=[]
            cands_image=[]
            cands_caption=[]
            for doc in cands[:self.topk]:
                    if isinstance(doc, dict):
                        cands_image.append(self.encode_img(doc['image_path']))
                        if 'image_caption' in doc:
                            cands_caption.append(doc['image_caption'])
                    else:
                        doc=doc.split()
                        doc=doc[:400]
                        doc=" ".join(doc)
                        cands_text.append(doc)
            if len(cands_text)!=0:
                instance['cands_text']=" ".join([f"[{i+1}] {sentence}" for i, sentence in enumerate(cands_text)])
            if len(cands_caption)!=0:
                instance['cands_caption']=" ".join([f"[{i+1}] {sentence}" for i, sentence in enumerate(cands_caption)])
            if len(cands_image)!=0:
                instance['cands_image']=cands_image

        return instance

