# for vanilla minicpmv
export CUDA_VISIBLE_DEVICES=0
python ../src/minicpmv/compute_ppl_minicpmv.py \
--model_path minicpmv-2_6 \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path ../output/retrieval/image_rerank/webqa_image_rerank_query_test_visualbge_5_image.trec \
--image_path ../output/retrieval/image_rerank/webqa_image_rerank_test_retrieval_images_5 \
--out_path ../output/image_rerank/test_minicpmv_ppl.json \
--topk 5

# for fine-tuned minicpmv
export CUDA_VISIBLE_DEVICES=0
python ../src/minicpmv/compute_ppl_minicpmv.py \
--model_path minicpmv-2_6 \
--path_to_adapter path/to/adapter \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path ../output/retrieval/image_rerank/webqa_image_rerank_query_test_visualbge_5_image.trec \
--image_path ../output/retrieval/image_rerank/webqa_image_rerank_test_retrieval_images_5 \
--out_path ../output/image_rerank/test_sft_minicpmv_ppl.json \
--topk 5