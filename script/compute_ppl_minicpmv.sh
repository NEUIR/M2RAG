# for vanilla minicpmv
export CUDA_VISIBLE_DEVICES=0
python ../src/minicpmv/compute_ppl_minicpmv.py \
--model_path minicpmv-2_6 \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path ../output/retrieval/image_rerank/webqa_image_rerank_query_test_visualbge_5_image.trec \
--image_path ../output/retrieval/image_rerank/webqa_image_rerank_test_retrieval_images_5 \
--out_path ../output/image_rerank/test_minicpmv_5_ppl.json \
--topk 5

python ../src/select_lowest_ppl_data.py \
--input_file ../output/image_rerank/test_minicpmv_5_ppl.json \
--output_file ../output/image_rerank/test_minicpmv_5_ppl_selected.jsonl \
--output_dir ../output/image_rerank/image_rerank_retrieve_minicpmv_ppl_top \
--restore_image \
--topk 5

# after restored images, you can use pytorch_fid to calculate the metric between the output_dir and the ground truth images dir


# for fine-tuned minicpmv
export CUDA_VISIBLE_DEVICES=0
python ../src/minicpmv/compute_ppl_minicpmv.py \
--model_path minicpmv-2_6 \
--path_to_adapter path/to/adapter \
--data_path ../data/m2rag/image_rerank/test_data.jsonl \
--trec_path ../output/retrieval/image_rerank/webqa_image_rerank_query_test_visualbge_5_image.trec \
--image_path ../output/retrieval/image_rerank/webqa_image_rerank_test_retrieval_images_5 \
--out_path ../output/image_rerank/test_sft_minicpmv_5_ppl.json \
--topk 5

python ../src/select_lowest_ppl_data.py \
--input_file ../output/image_rerank/test_sft_minicpmv_5_ppl.json \
--output_file ../output/image_rerank/test_sft_minicpmv_5_ppl_selected.jsonl \
--output_dir ../output/image_rerank/image_rerank_retrieve_sft_minicpmv_ppl_top \
--restore_image \
--topk 5

# after restored images, you can use pytorch_fid to calculate the metric between the output_dir and the ground truth images dir
