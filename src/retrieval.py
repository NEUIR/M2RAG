import sys
import csv
from tqdm import tqdm
import collections
import gzip
import pickle
import faiss
import os
import logging
import argparse
import json
import os.path as op
import numpy as np
logger = logging.getLogger()
import random


def convert_to_string_id(result_dict):
    string_id_dict = {}
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v
        string_id_dict[str(k)] = _temp_v
    return string_id_dict


def load_file(path):
    query_ids = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            query_ids[example['id']] = 1
    return query_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--query_embed_path",type=str,default='')
    parser.add_argument("--cand_image_embed_path",type=str,default='')
    parser.add_argument("--cand_text_embed_path",type=str,default='')
    parser.add_argument("--cand_multi_embed_path",type=str)
    
    parser.add_argument("--data_path", default='../data/m2rag/webqa/test_data.jsonl')
    parser.add_argument("--dataset_name", type=str, default='webqa')
    parser.add_argument("--out_path", default='../output/retrieval/')
    parser.add_argument('--retriever_name', type=str,default='visualbge')
    parser.add_argument('--task',type=str,default='image_cap') #{'image_cap', 'mmqa','fact_verify', 'image_rerank'}
    parser.add_argument('--flag', type=str, default='test')
    parser.add_argument('--retrieval_modality',default='multi') #{'text', 'image','multi'}

    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--topN", type=int, default=1)
    parser.add_argument('--use_all_gpus', action='store_true', default=False)


    args = parser.parse_args()

    args.output_dir=os.path.join(args.out_path,args.task)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    handlers = [logging.FileHandler(os.path.join(args.output_dir, 'evaluation_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    all_idx = []
    all_embeds = []
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(args.dim)

    if args.use_all_gpus:
        num_gpu = faiss.get_num_gpus()
        print('[faiss gpu] #GPU: {}'.format(num_gpu))
        co = faiss.GpuMultipleClonerOptions()
        # co.useFloat16 = True
        co.usePrecomputed = False
        co.shard = True

        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=num_gpu)
        # res = faiss.StandardGpuResources()
        # gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    
    if args.cand_image_embed_path and args.retrieval_modality!='text': 
        logger.info("load data from {}".format(args.cand_image_embed_path))
        with open(args.cand_image_embed_path, 'rb') as fin:
            doc_idx, doc_embeds = pickle.load(fin)
            if args.use_all_gpus:
                gpu_index.add(np.array(doc_embeds, dtype=np.float32))
            else:
                cpu_index.add(np.array(doc_embeds, dtype=np.float32))
            del doc_embeds
            all_idx.extend(doc_idx)
    
    if args.cand_text_embed_path and args.retrieval_modality!='image':
        logger.info("load data from {}".format(args.cand_text_embed_path))
        with open(args.cand_text_embed_path, 'rb') as fin:
            doc_idx, doc_embeds = pickle.load(fin)
            if args.use_all_gpus:
                gpu_index.add(np.array(doc_embeds, dtype=np.float32))
            else:
                cpu_index.add(np.array(doc_embeds, dtype=np.float32))
            del doc_embeds
            all_idx.extend(doc_idx)
    
    
    if args.cand_multi_embed_path and args.retrieval_modality =='multi': # used for image_cap cands
        logger.info("load data from {}".format(args.cand_multi_embed_path))
        with open(args.cand_multi_embed_path, 'rb') as fin:
            doc_idx, doc_embeds = pickle.load(fin)
            if args.use_all_gpus:
                gpu_index.add(np.array(doc_embeds, dtype=np.float32))
            else:
                cpu_index.add(np.array(doc_embeds, dtype=np.float32))
            del doc_embeds
            all_idx.extend(doc_idx)


    filted_qids = load_file(args.data_path)
    with open(args.query_embed_path, 'rb') as fin:
        logger.info("load data from {}".format(args.query_embed_path))
        query_idx, query_embeds = pickle.load(fin)
    
    all_embeds = np.array(all_embeds, dtype=np.float32)
    logger.info("Retrieve candidates!")
    query_embeds = np.array(query_embeds, dtype=np.float32)
    if args.use_all_gpus:
        D, I = gpu_index.search(query_embeds, args.topN)
    else:
        D, I = cpu_index.search(query_embeds, args.topN)
    ctx_idxs = {}
    assert len(query_idx) == len(I)
    for step, qid in enumerate(query_idx):
        if qid in filted_qids:
            ctx_idxs[qid] = []
            for idx in I[step]:
                ctx_idxs[qid].append(str(all_idx[idx]))

    trec_path = os.path.join(args.output_dir, f'{args.dataset_name}_{args.task}_query_{args.flag}_{args.retriever_name}_{args.topN}_{args.retrieval_modality}.trec')
    results_data = []
    assert len(query_idx) == len(I)
    for step, qid in enumerate(query_idx):
        if qid in filted_qids:
            for idx, score in zip(I[step], D[step]):
                    results_data.append([qid, all_idx[idx], str(score)])
                    
    with open(trec_path, 'w') as fout:
        for item in results_data:
            fout.write("\t".join(map(str, item)) + "\n")