# for vanilla Qwen2VL
export CUDA_VISIBLE_DEVICES=0
python ../src/qwen_vl/compute_ppl_qwen2vl.py \
--model_path path/to/Qwen2-VL-7B-Instruct \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path path/to/retrieve_results.trec \
--image_path path/to/images \
--out_path ../output/test_qwen2vl_5_ppl.json \
--topk 5

# for fine-tuned Qwen2VL
export CUDA_VISIBLE_DEVICES=0
python ../src/qwen_vl/compute_ppl_qwen2vl.py \
--model_path path/to/fine_tuned_model \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path path/to/retrieve_results.trec \
--image_path path/to/images \
--out_path ../output/test_sft_qwen2vl_5_ppl.json \
--topk 5