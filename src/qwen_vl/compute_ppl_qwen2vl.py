import argparse
import json
import logging
import os
from functools import partial
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn as nn

import torch
import transformers

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor
from qwen2vl_datasets_modules_for_ppl import make_supervised_data_module, DataArguments

def load_dataset(processor, args):
    data_module = make_supervised_data_module(processor=processor, data_args=args)
    sampler = SequentialSampler(data_module['train_dataset'])
    reader = DataLoader(dataset=data_module['train_dataset'], sampler=sampler, num_workers=args.num_workers, batch_size=args.batch_size,
                        collate_fn=data_module['data_collator'], drop_last=False)
    return reader, len(data_module['train_dataset'])


def load_model_tokenizer(model_path, args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True,
                                        attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_path, max_pixels=args.max_pixels)
    return model, processor

def move_inputs_to_device(inputs, device):
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['labels'] = inputs['labels'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(device)
    inputs['image_grid_thw'] = inputs['image_grid_thw'].to(device)

    return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",type=str,default='')
    parser.add_argument('--data_path',type=str,default='')
    parser.add_argument('--trec_path',type=str,default='')
    parser.add_argument('--image_path',type=str,default='')
    parser.add_argument("--out_path", type=str, default='')
    parser.add_argument('--retriever_name',type=str,default='_')

    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--num_workers',type=int,default=1)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--max_new_tokens',type=int,default=64)

    parser.add_argument('--LLM_TYPE', type=str, default='qwen2')
    parser.add_argument('--max_slice_nums', type=int, default=9)
    parser.add_argument('--model_max_length', type=int, default=2048)
    parser.add_argument('--topk', type=int, default=5)  #[0,5]

    args = parser.parse_args()
    data_args = DataArguments()
    args.__dict__.update(data_args.__dict__)
    args.image_folder = ""

    model, processor = load_model_tokenizer(args.model_path, args)
    model = model.cuda('cuda:0')
    data_reader, total = load_dataset(processor, args)
    ppl_list = []
    for step, inputs in tqdm(enumerate(data_reader), total=total):
        with torch.no_grad():
            inputs = move_inputs_to_device(inputs, model.device)
            if "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            if "id" in inputs and "image_path" in inputs:
                id = inputs.pop("id")[-1]
                image_path = inputs.pop("image_path")[-1]
            else:
                id = None
                image_path = None
            if 'is_dummy' in inputs:
                inputs.pop('is_dummy')
            outputs = model(**inputs, use_cache=False)
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1,
                                        model.config.vocab_size).contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1).long().contiguous()
            loss = loss_fct(shift_logits, shift_labels)
            ppl = torch.exp(loss)
            ppl_list.append((id,image_path,ppl.item()))
    with open(args.out_path, 'w') as f:
        json.dump(ppl_list, f)

