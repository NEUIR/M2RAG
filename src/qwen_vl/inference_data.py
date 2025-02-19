from qwen_vl_utils import process_vision_info
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

class InferenceDataset(Dataset):
    def __init__(self, args, data, prompt,processor=None):
        
        self.data=data
        self.prompt=prompt
        self.processor=processor
        self.topk = args.topk
        self.device=args.device
        self.task=args.task  # {'mmqa','fact_verify', 'image_cap'}
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
                    instruction,input=self.prompt.split('<image>\n')
                    content=[{"type": "text", "text":instruction}, {"type": "image", "image": example['image']}]

                    if 'cands_image' in example:
                        images=[{"type": "image", "image":image} for image in example['cands_image']]
                        content.extend(images)
        
                    # retrirval_image_caption=example['cands_caption'] if 'cands_caption' in example else 'null'
                    if 'cands_caption' in example:
                        retrirval_image_caption=example['cands_caption']
                    elif 'cands_text' in example:
                        retrirval_image_caption=example['cands_text']
                    else:
                        retrirval_image_caption='null'
                    
                    input=input.format(retrirval_image_caption=retrirval_image_caption)
                    content.append({"type": "text", "text":input})
                    
                elif self.task=='mmqa':
                    instruction,input=self.prompt.split('<image>\n')
                    content=[{"type": "text", "text":instruction}]
                    retrirval_text=example['cands_text'] if 'cands_text' in example else 'null'
                    if 'cands_image' in example:
                        content_images=[{"type": "image", "image":image} for image in example['cands_image']]
                        content.extend(content_images)
                    retrirval_image_caption=example['cands_caption'] if 'cands_caption' in example else 'null'
                    input=input.format(question=example['query'], retrirval_text=retrirval_text, retrirval_image_caption=retrirval_image_caption)
                    content.append({"type": "text", "text":input})
                        
                elif self.task=='fact_verify':
                    instruction,input=self.prompt.split('<image>\n')
                    content=[{"type": "text", "text":instruction}, {"type": "image", "image": example['claim_image']}]
                    retrirval_text=example['cands_text'] if 'cands_text' in example else 'null'
                    if 'cands_image' in example:
                        content_images=[{"type": "image", "image":image} for image in example['cands_image']]
                        content.extend(content_images)
                    input=input.format(claim_text=example['claim'], retrirval_text=retrirval_text)
                    content.append({"type": "text", "text":input})

                else:
                    raise ValueError("The task is error!")
                
                msg=[{"role": "user", "content": content}]
                
            else:
                if self.task=='image_cap':
                    instruction,input=self.prompt.split('<image>\n')
                    msg=[{"role": "user", 
                        "content": [
                            {"type": "text", "text":instruction},
                            {"type": "image", "image":example['image']},
                            {"type": "text", "text":input},
                        ]}]
                    
                elif self.task=='mmqa':
                    msg=[{"role": "user", 
                        "content": [
                            {"type": "text", "text":self.prompt.format(question=example['query'])},
                        ]}]
                    
                elif self.task=='fact_verify':
                    instruction,input=self.prompt.split('<image>\n')

                    msg=[{"role": "user", 
                        "content": [
                            {"type": "text", "text":instruction},
                            {"type": "image", "image":example['claim_image']},
                            {"type": "text", "text":input.format(claim_text=example['claim'])},
                        ]}]
                else:
                    raise ValueError("The task is error!")
                
            msgs_list.append(msg)
            
        texts = [
                    self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in msgs_list
                ]
        image_inputs, video_inputs = process_vision_info(msgs_list)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        processed_batch['inputs']=inputs
        processed_batch['qids']=qids_list
        return processed_batch


    def __getitem__(self, index):
        example = self.data[index]
        instance = {}
        instance['id'] = example['id']
        if self.task=='image_cap':
            instance['image']=example['image_path']
            
        elif self.task=='mmqa':
            instance['query']=example['query']
            
        elif self.task=='fact_verify':
            instance['claim']=example['claim']
            instance['claim_image']=example['claim_image']
            
            document=example['document'].split()
            document=document[:500]
            document=' '.join(document)
            instance['document']=document
            
            instance['document_image']=example['document_image']

        else:
            raise ValueError("The task is error!")

        if self.topk!=0:
            cands=self.query_caption_map[str(example['id'])]
            
            cands_text=[]
            cands_image=[]
            cands_caption=[]
            for doc in cands[:self.topk]:
                    if isinstance(doc, dict):
                        cands_image.append(doc['image_path'])
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

