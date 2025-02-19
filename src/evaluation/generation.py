import json
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider_scorer import CiderScorer

# 加载 JSONL 文件
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 计算 BLEU (1-4)
def calculate_bleu(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    smoothie = SmoothingFunction().method1
    bleu_scores = {
        f"BLEU-{i}": sentence_bleu(reference_tokens, candidate_tokens, weights=(1/i,) * i, smoothing_function=smoothie)
        for i in range(1, 5)
    }
    return bleu_scores

# 计算 ROUGE
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {k: v.fmeasure for k, v in scores.items()}

# 计算 CIDEr
def calculate_cider(references, candidates):
    scorer = CiderScorer(n=4, sigma=6.0)
    for ref, cand in zip(references, candidates):
        scorer += (cand, [ref])
    cider, _ = scorer.compute_score()
    return cider

# 主程序
def main(reference_file, candidate_file):
    # 加载 JSONL 文件
    references = load_jsonl(reference_file)
    candidates = load_jsonl(candidate_file)
    
    if len(references) != len(candidates):
        raise ValueError("The number of references and candidates must be the same.")
        # references=references[:len(candidates)]
    #     references=references[:300]
    #     candidates=candidates[:300]
    # else:
    #     references=references[:2000]
    #     candidates=candidates[:2000]
        # 

    total_bleu = {f"BLEU-{i}": 0.0 for i in range(1, 5)}
    total_rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    reference_texts = []
    candidate_texts = []

    for ref, cand in zip(references, candidates):
        if ref["id"] != cand["id"]:
            raise ValueError(f"ID mismatch: {ref['id']} != {cand['id']}")
        
        if 'answer' in ref and ref['answer']:
            reference = ref['answer'][0]
        elif 'caption' in ref and ref['caption']:
            reference = ref['caption']
        elif 'clean_page_description' in ref and ref['clean_page_description']:
            reference = ref['clean_page_description']
        elif 'context_text' in ref and ref['context_text']:
            reference = ref['context_text']

        candidate = cand["text"] if 'text' in cand else cand['generated_caption']
        
        # compute for BLEU
        bleu_scores = calculate_bleu(reference, candidate)
        for k, v in bleu_scores.items():
            total_bleu[k] += v
        
        # compute for ROUGE
        rouge_scores = calculate_rouge(reference, candidate)
        for k, v in rouge_scores.items():
            total_rouge[k] += v
        
        # collect texts for compute CIDEr
        reference_texts.append(reference)
        candidate_texts.append(candidate)
    
    # average score for BLEU 和 ROUGE
    num_samples = len(references)
    avg_bleu = {k: v / num_samples for k, v in total_bleu.items()}
    avg_rouge = {k: v / num_samples for k, v in total_rouge.items()}

    # compute for CIDEr
    avg_cider = calculate_cider(reference_texts, candidate_texts)
    
    # print results
    print("Average BLEU Scores:")
    for k, v in avg_bleu.items():
        print(f"{k}: {v:.4f}")
    
    print("\nAverage ROUGE Scores:")
    for k, v in avg_rouge.items():
        print(f"{k}: {v:.4f}")
    
    print(f"\nCIDEr Score: {avg_cider:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--reference_file",type=str)
    parser.add_argument("--candidate_file", type=str)
    args = parser.parse_args()

    main(args.reference_file, args.candidate_file)