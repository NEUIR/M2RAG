import json
from typing import DefaultDict
import jsonlines
import random
random.seed(2025)
import os
import argparse
import sys
from data.finetune.get_prompt_new import get_prompt

# image caption task
def load_data(data_path):
    data_dict = {}
    with jsonlines.open(data_path, 'r') as reader:
        for item in reader:
            data_dict[item['id']] = item
    return data_dict


def merge_origin_data_with_retrieved_data(origin_data: dict,retrieved_data: dict, task: str):
    assert task in ['mmqa','fact_verify','wiki_gen','image_cap']
    data_list = []
    assert len(list(set(origin_data.keys()))) == len(list(set(retrieved_data.keys())))
    assert len(origin_data) == len(retrieved_data)
    for key in origin_data.keys():
        one_data = origin_data[key]
        one_data['cands'] = retrieved_data[key]['cands']
        data_list.append(one_data)
    return data_list

def transform_to_conversation_type(data_list, task, llm_type,topk):
    assert task in ['mmqa','fact_verify','wiki_gen','image_cap']
    conversation_data_list = []
    if task == 'image_cap':
        for data in data_list:
            # collect data
            main_image = data['image_path']
            main_image_caption = data['caption']
            id = task + "_" + data['id']
            retrieval_image_list = []
            retrieval_image_caption_list = []
            retrieval_text_list = []
            # topk = len(data['cands'])
            for retrieved_cand in data['cands'][:topk]:
                if isinstance(retrieved_cand,str):
                    retrieved_cand = retrieved_cand.split()
                    retrieved_cand = retrieved_cand[:400]
                    retrieved_cand = ' '.join(retrieved_cand)
                    retrieval_text_list.append(retrieved_cand)
                elif isinstance(retrieved_cand,dict):
                    if args.retrieval_modality != 'text':
                        retrieval_image_list.append(retrieved_cand['image_path'])
                    if args.retrieval_modality != 'image':
                        retrieval_image_caption_list.append(retrieved_cand['image_caption'])
                else:
                    raise NotImplementedError
            
            # construct data
            # 1、 construct image template
            if len(retrieval_image_list)!=0:
                image_list = [main_image] + retrieval_image_list
            else:
                image_list = [main_image]
            if llm_type == 'minicpmv':
                image_template_dict, image_template = construct_image_template_for_minicpmv(image_list)
            elif llm_type == 'qwen2vl':
                #TODO:
                image_template_dict, image_template = construct_image_template_for_qwen2vl(image_list)
            else:
                raise NotImplementedError
            # 2、 construct image caption and text
            retrieval_image_caption_str = 'null'
            if len(retrieval_image_caption_list)!= 0:
                retrieval_image_caption_str = ' '.join(f'[{i+1}] {cap}' for i, cap in enumerate(retrieval_image_caption_list))
            # 3、 construct prompt template
            prompt = get_prompt(task=task, topk=topk)
            prompt = prompt.replace("Image: The first image.",
                                    f"{image_template}\nImage: The first image.")
            prompt = prompt.format(retrieval_image_caption = retrieval_image_caption_str)
            # 4、 construct conversation
            if llm_type == 'minicpmv':
                conversations = [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": main_image_caption
                    }
                ]
                # 5、 construct data
                one_conversation_data = {
                    "id": id,
                    "image": image_template_dict,
                    "conversations": conversations
                }
                conversation_data_list.append(one_conversation_data)
            elif llm_type == 'qwen2vl':
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": main_image_caption
                    }
                ]
                # 5、 construct data
                one_conversation_data = {
                    "messages": messages,
                    "images": image_template_dict
                }
                conversation_data_list.append(one_conversation_data)
    elif task == 'mmqa':
        for data in data_list:
            # collect data
            main_query = data['query']
            main_answer = data['answer'][0]
            id = task + "_" + data['id']
            retrieval_image_list = []
            retrieval_image_caption_list = []
            retrieval_text_list = []
            # topk = len(data['cands'])
            for retrieved_cand in data['cands'][:topk]:
                if isinstance(retrieved_cand,str):
                    retrieved_cand = retrieved_cand.split()
                    retrieved_cand = retrieved_cand[:400]
                    retrieved_cand = ' '.join(retrieved_cand)
                    retrieval_text_list.append(retrieved_cand)
                elif isinstance(retrieved_cand,dict):
                    retrieval_image_list.append(retrieved_cand['image_path'])
                    retrieval_image_caption_list.append(retrieved_cand['image_caption'])
                else:
                    raise NotImplementedError
            # construct data
            # 1、 construct image template
            image_list = retrieval_image_list
            if llm_type =='minicpmv':
                image_template_dict, image_template = construct_image_template_for_minicpmv(image_list)
            elif llm_type == 'qwen2vl':
                #TODO:
                image_template_dict, image_template = construct_image_template_for_qwen2vl(image_list)
            else:
                raise NotImplementedError
            # 2、 construct image caption and text
            retrieval_image_caption_str = 'null'
            if len(retrieval_image_caption_list)!= 0:
                retrieval_image_caption_str =' '.join(f'[{i+1}] {cap}' for i, cap in enumerate(retrieval_image_caption_list))
            retrieved_text_str = 'null'
            if len(retrieval_text_list)!= 0:
                retrieved_text_str =' '.join(f'[{i+1}] {text}' for i, text in enumerate(retrieval_text_list))
            # 3、 construct prompt template
            prompt = get_prompt(task=task, topk=topk)
            prompt = prompt.replace("Question:",
                                    f"{image_template}\nQuestion:")
            prompt = prompt.format(question = main_query,
                                   retrieval_image_caption = retrieval_image_caption_str,
                                   retrieval_text = retrieved_text_str)
            # 4、 construct conversation
            if llm_type =='minicpmv':
                conversations = [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": main_answer
                    }
                ]
                # 5、 construct data
                one_conversation_data = {
                    "id": id,
                    "image": image_template_dict,
                    "conversations": conversations
                }
                conversation_data_list.append(one_conversation_data)
            elif llm_type == 'qwen2vl':
                messages  = [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": main_answer
                    }
                ]
                # 5、 construct data
                one_conversation_data = {
                    "messages": messages,
                    "images": image_template_dict
                }
                conversation_data_list.append(one_conversation_data)
    elif task == 'fact_verify':
        for data in data_list:
            # collect data
            main_claim_text = data['claim']
            main_claim_image = data['claim_image']
            main_doc_text = data['document']
            main_doc_text = main_doc_text.split()
            main_doc_text = main_doc_text[:500]
            main_doc_text = ' '.join(main_doc_text)
            main_doc_image = data['document_image']
            main_category = data['category']
            id = task + "_" + data['id']
            retrieval_image_list = []
            retrieval_text_list = []
            # topk = len(data['cands'])
            for retrieved_cand in data['cands'][:topk]:
                if isinstance(retrieved_cand,str):
                    retrieved_cand = retrieved_cand.split()
                    retrieved_cand = retrieved_cand[:400]
                    retrieved_cand =' '.join(retrieved_cand)
                    retrieval_text_list.append(retrieved_cand)
                elif isinstance(retrieved_cand,dict):
                    retrieval_image_list.append(retrieved_cand['image_path'])
                else:
                    raise NotImplementedError
            # construct data
            # 1、 construct image template
            image_list = [main_claim_image, main_doc_image] + retrieval_image_list
            if llm_type =='minicpmv':
                image_template_dict, image_template = construct_image_template_for_minicpmv(image_list)
            elif llm_type == 'qwen2vl':
                #TODO:
                image_template_dict, image_template = construct_image_template_for_qwen2vl(image_list)
            else:
                raise NotImplementedError
            # 2、 construct text
            retrieval_text_str = 'null'
            if len(retrieval_text_list)!= 0:
                retrieval_text_str =' '.join(f'[{i+1}] {text}' for i, text in enumerate(retrieval_text_list))
            # 3、 construct prompt template
            prompt = get_prompt(task=task, topk=topk)
            prompt = prompt.replace("Claim_Image: The first image.",
                                    f"{image_template}\nClaim_Image: The first image.")
            prompt = prompt.format(claim_text = main_claim_text,doc_text=main_doc_text, retrieval_text = retrieval_text_str)
            # 4、 construct conversation
            if llm_type =='minicpmv':
                conversations = [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": main_category
                    }
                ]
                # 5、 construct data
                one_conversation_data = {
                    "id": id,
                    "image": image_template_dict,
                    "conversations": conversations
                }
                conversation_data_list.append(one_conversation_data)
            elif llm_type == 'qwen2vl':
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": main_category
                    }
                ]
                # 5、 construct data
                one_conversation_data = {
                    "messages": messages,
                    "images": image_template_dict
                }
                conversation_data_list.append(one_conversation_data)
    elif task == 'wiki_gen':
        for data in data_list:
            # collect data
            main_image = data['image_path']
            main_image_title = data['page_title']
            main_image_page_description = data['clean_page_description']
            id = task + "_" + data['id']
            retrieval_image_list = []
            retrieval_text_list = []
            # topk = len(data['cands'])
            for retrieved_cand in data['cands'][:topk]:
                if isinstance(retrieved_cand,str):
                    retrieved_cand = retrieved_cand.split()
                    retrieved_cand = retrieved_cand[:400]
                    retrieved_cand =' '.join(retrieved_cand)
                elif isinstance(retrieved_cand,dict):
                    retrieval_image_list.append(retrieved_cand['image_path'])
                else:
                    raise NotImplementedError
            # construct data
            # 1、 construct image template
            image_list = [main_image] + retrieval_image_list
            if llm_type =='minicpmv':
                image_template_dict, image_template = construct_image_template_for_minicpmv(image_list)
            elif llm_type == 'qwen2vl':
                #TODO:
                image_template_dict, image_template = construct_image_template_for_qwen2vl(image_list)
            else:
                raise NotImplementedError
            # 2、 construct text
            retrieved_text_str = 'null'
            if len(retrieval_text_list) != 0:
                retrieved_text_str =''.join(f'[{i+1}] {text}' for i, text in enumerate(retrieval_text_list))
            # 3、 construct prompt template
            prompt = get_prompt(task=task, topk=topk)
            prompt = prompt.replace("Image: The first image.",
                                    f"{image_template}\nImage: The first image.")
            prompt = prompt.format(title = main_image_title,
                                   retrieval_text = retrieved_text_str)
            # 4、 construct conversation
            if llm_type =='minicpmv':
                conversations = [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": main_image_page_description
                    }
                ]
                # 5、 construct data
                one_conversation_data = {
                    "id": id,
                    "image": image_template_dict,
                    "conversations": conversations
                }
                conversation_data_list.append(one_conversation_data)
            elif llm_type == 'qwen2vl':
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": main_image_page_description
                    }
                ]
                # 5、 construct data
                one_conversation_data = {
                    "messages": messages,
                    "images": image_template_dict
                }
                conversation_data_list.append(one_conversation_data)
    else:
        raise NotImplementedError
    
    return conversation_data_list

def construct_image_template_for_minicpmv(image_list):
    if len(image_list) == 1:
        image_template = "<image>\n"
        template_image_dict = image_list[0]
        return template_image_dict, image_template
    else:
        template_image_dict = {}
        image_template = ''
        for i,image_path in enumerate(image_list):
            template_image_dict[f'<image_0{i}>'] = image_path
            image_template += f'<image_0{i}>\n'
        return template_image_dict, image_template

def construct_image_template_for_qwen2vl(image_list):
    template_image_dict = []
    image_template = ''
    for i, image_path in enumerate(image_list):
        template_image_dict.append(image_path)
        image_template += f'<image>\n'
    return template_image_dict, image_template

def split_val_set_for_minicpmv(conversation_data_list):
    # train有3000条，从3000中随机采样抽取100条作为val
    val_list = []
    train_list = []
    data_idx = range(len(conversation_data_list))
    val_data_idx = random.sample(data_idx,k=100)
    for idx, data in enumerate(conversation_data_list):
        if idx in val_data_idx:
            val_list.append(data)
        else:
            train_list.append(data)
    return train_list, val_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # # multi
    parser.add_argument("--mmqa_data", type=str, default='')
    parser.add_argument("--mmqa_retrieve_data", type=str, default='')
    parser.add_argument("--fact_verify_data", type=str, default='')
    parser.add_argument("--fact_verify_retrieve_data", type=str, default='')
    parser.add_argument("--image_cap_data", type=str, default='')
    parser.add_argument("--image_cap_retrieve_data", type=str, default='')

    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--llm_type", type=str, default='')
    parser.add_argument("--topk", type=int, default=5)
    

    
    args = parser.parse_args()

    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1、 load data
    mmqa_data = load_data(args.mmqa_data)
    mmqa_retrieve_data = load_data(args.mmqa_retrieve_data)
    
    fact_verify_data = load_data(args.fact_verify_data)
    fact_verify_retrieve_data = load_data(args.fact_verify_retrieve_data)
    
    image_cap_data = load_data(args.image_cap_data)
    image_cap_retrieve_data = load_data(args.image_cap_retrieve_data)
    # 2、 merge data
    mmqa_data_list = merge_origin_data_with_retrieved_data(mmqa_data, mmqa_retrieve_data, task='mmqa')
    fact_verify_data_list = merge_origin_data_with_retrieved_data(fact_verify_data, fact_verify_retrieve_data, task='fact_verify')
    image_cap_data_list = merge_origin_data_with_retrieved_data(image_cap_data, image_cap_retrieve_data, task='image_cap')
    # 3、 transform to conversation type
    mmqa_conversation_data_list = transform_to_conversation_type(mmqa_data_list, task='mmqa', llm_type=args.llm_type,topk=args.topk)
    fact_verify_conversation_data_list = transform_to_conversation_type(fact_verify_data_list, task='fact_verify', llm_type=args.llm_type,topk=args.topk)
    image_cap_conversation_data_list = transform_to_conversation_type(image_cap_data_list, task='image_cap', llm_type=args.llm_type,topk=args.topk)
    # save data
    if args.llm_type == 'minicpmv':
        mmqa_train_list, mmqa_val_list = split_val_set_for_minicpmv(mmqa_conversation_data_list)
        fact_verify_train_list, fact_verify_val_list = split_val_set_for_minicpmv(fact_verify_conversation_data_list)
        image_cap_train_list, image_cap_val_list = split_val_set_for_minicpmv(image_cap_conversation_data_list)
        train_list = mmqa_train_list + fact_verify_train_list  + image_cap_train_list
        val_list = mmqa_val_list + fact_verify_val_list  + image_cap_val_list
        random.shuffle(train_list)
        random.shuffle(val_list)
        # save data
        train_file_name = 'minicpmv_supervised_train.json'
        val_file_name = 'minicpmv_supervised_val.json'
        with open(os.path.join(args.output_dir, train_file_name), 'w') as f:
            f.write(json.dumps(train_list, indent=4))
        with open(os.path.join(args.output_dir, val_file_name), 'w') as f:
            f.write(json.dumps(val_list, indent=4))
    elif args.llm_type == 'qwen2vl':
        train_list = mmqa_conversation_data_list + fact_verify_conversation_data_list + image_cap_conversation_data_list
        random.shuffle(train_list)
        train_file_name ='qwen2vl_supervised_train.json'
        with open(os.path.join(args.output_dir, train_file_name), 'w') as f:
            f.write(json.dumps(train_list, indent=4))
