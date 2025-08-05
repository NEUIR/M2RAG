import base64
import json
import io
import torch
from torch.utils.data import Dataset
import os
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

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

        if hasattr(args, 'doc_modality'):
            assert args.doc_modality in ['multi', 'text', 'image']
            self.doc_modality = args.doc_modality
        else:
            self.doc_modality = 'multi'


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
                if self.task=='mmqa':
                    instruction,input=self.prompt.split('<image>\n')
                    content=[{"type": "text", "text":instruction}]
                    retrirval_text=example['cands_text'] if 'cands_text' in example else 'null'
                    if 'cands_image' in example:
                        content_image=[{"type": "image", "image":image} for image in example['cands_image']]
                        content.extend(content_image)
                    retrirval_image_caption=example['cands_caption'] if 'cands_caption' in example else 'null'
                    input=input.format(question=example['query'], retrirval_text=retrirval_text, retrirval_image_caption=retrirval_image_caption)
                    content.append({"type": "text", "text":input})
                        
                elif self.task=='fact_verify':
                    instruction,input=self.prompt.split('<image>\n')
                    content=[{"type": "text", "text":instruction}]
                    content=content+[
                            {"type": "image", "image":example['claim_image']}]
                    retrirval_text=example['cands_text'] if 'cands_text' in example else 'null'
                    if 'cands_image' in example:
                        images=[{"type": "image", "image":image} for image in example['cands_image']]
                        content.extend(images)
                    input=input.format(claim_text=example['claim'], retrirval_text=retrirval_text)
                    content.append({"type": "text", "text":input})
                    
                elif self.task=='image_cap':
                    instruction,input=self.prompt.split('<image>\n')
                    content=[{"type": "text", "text":instruction}]
                    content=content+[
                            {"type": "image", "image":example['image']}]
                    # retrirval_text=example['cands_text'] if 'cands_text' in example else 'null'
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
                # content.append({"type": "text", "text":text_prompt})
                
                msg=[{"role": "user", 
                    "content": content}]
            else:
                if self.task=='mmqa':
                    # {"type": "image", "image":example['image']},
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
                elif self.task=='image_cap':
                    instruction,input=self.prompt.split('<image>\n')
                    msg=[{"role": "user", 
                        "content": [
                            {"type": "text", "text":instruction},
                            {"type": "image", "image":example['image']},
                            {"type": "text", "text":input},
                        ]}]
            msgs_list.append(msg)
        inputs = self.processor.apply_chat_template(msgs_list, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to("cuda", dtype=torch.bfloat16)
        
        processed_batch['inputs']=inputs
        processed_batch['qids']=qids_list
        return processed_batch


    def __getitem__(self, index):
        example = self.data[index]
        instance = {}
        instance['id'] = example['id']
        if self.task=='mmqa':
            # instance['image']=example['image_path']
            instance['query']=example['query']
        elif self.task=='fact_verify':
            instance['claim']=example['claim']
            instance['claim_image']=example['claim_image']
            
            document=example['document'].split()
            document=document[:500]
            document=' '.join(document)
            instance['document']=document
            
            instance['document_image']=example['document_image']
        elif self.task=='image_cap':
            instance['image']=example['image_path']
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
            
            # added to ablation
            if self.doc_modality == 'multi':
                pass
            elif self.doc_modality == 'image':
                POP_FAIL_SIGNAL = 'POP_FAIE'
                instance.pop('cands_text', POP_FAIL_SIGNAL)
                instance.pop('cands_caption', POP_FAIL_SIGNAL)
                assert 'cands_text' not in instance and 'cands_caption' not in instance
            elif self.doc_modality == 'text':
                POP_FAIL_SIGNAL = 'POP_FAIE'
                instance.pop('cands_image', POP_FAIL_SIGNAL)
                instance['cands_image'] = []
                assert len(instance['cands_image']) == 0
            else:
                raise ValueError(f'Unknown doc_modality: {self.doc_modality}')
        return instance

