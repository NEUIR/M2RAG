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

from transformers import AutoModel, AutoTokenizer
from minicpmv_datasets_modules_for_ppl import load_data, SupervisedDataset, build_transform, data_collator

from peft import PeftModel


def load_model_tokenizer(args):
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                        attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    if args.path_to_adapter != None:
        model = PeftModel.from_pretrained(model,
                                          args.path_to_adapter,
                                          device_map="auto",
                                          trust_remote_code=True
                                          )
        print("loaded adapterd")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    return model, tokenizer

def load_dataset(args,tokenizer,model):
    raw_data = load_data(args.image_path,args.trec_path,args.data_path, topk=args.topk)
    transforms_func = build_transform()
    collator = partial(data_collator, max_length=args.model_max_length)

    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False
    train_dataset = SupervisedDataset(
        raw_data=raw_data,
        transform=transforms_func,
        tokenizer=tokenizer,
        slice_config=slice_config,
        llm_type=args.LLM_TYPE,
        patch_size=model.config.patch_size,
        query_nums=model.config.query_num,
        batch_vision=batch_vision,
        max_length=args.model_max_length
    )
    sampler = SequentialSampler(train_dataset)
    reader = DataLoader(dataset=train_dataset, sampler=sampler, num_workers=args.num_workers, batch_size=args.batch_size, 
                        collate_fn=collator, drop_last=False)
    return reader

def move_inputs_to_device(inputs, device):
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['position_ids'] = inputs['position_ids'].to(device)
    inputs['labels'] = inputs['labels'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    inputs['image_bound'] = [i.to(device) for i in inputs['image_bound']]
    inputs['tgt_sizes'] = [i.to(device) for i in inputs['tgt_sizes']]
    inputs['pixel_values'] = [[j.to(device) for j in i] for i in inputs['pixel_values']]

    return inputs



if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
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
    parser.add_argument('--path_to_adapter', type=str, default=None)    # for fine-tuned model, please add the path to the adapter

    args = parser.parse_args()

    model, tokenizer = load_model_tokenizer(args=args)
    model = model.eval().cuda('cuda')
    data_reader = load_dataset(args, tokenizer, model)
    ppl_list = []
    for step, inputs in tqdm(enumerate(data_reader)):
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
            
            outputs = model(data = inputs, use_cache=False)
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits.view(-1,
                                         model.config.vocab_size).contiguous()
            labels = labels.view(-1).long().contiguous()
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels).to('cpu')
            ppl = torch.exp(loss)
            ppl_list.append((id, image_path, ppl.item()))

    with open(args.out_path, 'w') as fout:
        fout.write(json.dumps(ppl_list))


