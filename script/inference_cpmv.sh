#

export CUDA_VISIBLE_DEVICES=0

# you can change the topk (0,1,3,5) to control the number of retrieved multi-modal documents used during the inference process

# ---------Task-1: image_cap----------
python ../src/minicpmv/inference.py \
--model_path openbmb/MiniCPM-V-2_6 \
--data_path ../data/m2rag/image_cap/test_data.jsonl \
--retrieval_data_path ../output/retrieval/image_cap/webqa_image_cap_test_retrieval_multi_5.jsonl \
--out_path ../output/gen_overall \
--topk 0 \
--task image_cap \
--dataset_name webqa \
--max_new_tokens 64 \
--batch_size 4

# ---------Task-2: mmqa----------
python ../src/minicpmv/inference.py \
--model_path openbmb/MiniCPM-V-2_6 \
--data_path ../data/m2rag/mmqa/train_data.jsonl \
--retrieval_data_path ../output/retrieval/mmqa/webqa_mmqa_test_retrieval_multi_5.jsonl \
--out_path ../output/gen_overall \
--topk 1 \
--task mmqa \
--dataset_name webqa \
--max_new_tokens 64 \
--batch_size 4

# ---------Task-3: fact_verify----------
python ../src/minicpmv/inference.py \
--model_path openbmb/MiniCPM-V-2_6 \
--data_path ../data/m2rag/fact_verify/val_data.jsonl \
--retrieval_data_path ../output/retrieval/fact_verify/factify_fact_verify_test_retrieval_multi_5.jsonl \
--out_path ../output/gen_overall \
--topk 3 \
--task fact_verify \
--dataset_name factify \
--max_new_tokens 4 \
--batch_size 1
