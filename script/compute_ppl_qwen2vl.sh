# for vanilla Qwen2VL
export CUDA_VISIBLE_DEVICES=0
python ../src/qwen_vl/compute_ppl_qwen2vl.py \
--model_path path/to/Qwen2-VL-7B-Instruct \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path ../output/retrieval/image_rerank/webqa_image_rerank_query_test_visualbge_5_image.trec \
--image_path ../output/retrieval/image_rerank/webqa_image_rerank_test_retrieval_images_5 \
--out_path ../output/test_qwen2vl_5_ppl.json \
--topk 5

# for fine-tuned Qwen2VL
export CUDA_VISIBLE_DEVICES=0
python ../src/qwen_vl/compute_ppl_qwen2vl.py \
--model_path path/to/fine_tuned_model \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path ../output/retrieval/image_rerank/webqa_image_rerank_query_test_visualbge_5_image.trec \
--image_path ../output/retrieval/image_rerank/webqa_image_rerank_test_retrieval_images_5 \
--out_path ../output/test_sft_qwen2vl_5_ppl.json \
--topk 5