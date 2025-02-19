# for vanilla minicpmv
export CUDA_VISIBLE_DEVICES=0
python ../src/minicpmv/compute_ppl_minicpmv.py \
--model_path minicpmv-2_6 \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path path/to/retrieve_results.trec \
--image_path path/to/images \
--out_path ../output/image_rerank/test_minicpmv_ppl.json \
--topk 5

# for fine-tuned minicpmv
export CUDA_VISIBLE_DEVICES=0
python ../src/minicpmv/compute_ppl_minicpmv.py \
--model_path minicpmv-2_6 \
--path_to_adapter path/to/adapter \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path path/to/retrieve_results.trec \
--image_path path/to/images \
--out_path ../output/image_rerank/test_sft_minicpmv_ppl.json \
--topk 5