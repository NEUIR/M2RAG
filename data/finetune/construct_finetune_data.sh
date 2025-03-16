# qwen-vl
python mix_training_data_new.py \
--mmqa_data data/m2rag/mmqa/train_data.jsonl \
--mmqa_retrieve_data output/retrieval/mmqa/webqa_mmqa_train_retrieval_multi_5.jsonl \
--fact_verify_data data/m2rag/fact_verify/train_data.jsonl \
--fact_verify_retrieve_data output/retrieval/fact_verify/factify_fact_verify_train_retrieval_multi_5.jsonl  \
--image_cap_data data/m2rag/image_cap/train_data.jsonl \
--image_cap_retrieve_data output/retrieval/image_cap/webqa_image_cap_train_retrieval_multi_5.jsonl \
--output_dir src/qwen_vl/LLaMA-Factory/data \
--llm_type qwen2vl \
--topk 5

# minicpmv
python mix_training_data_new.py \
--mmqa_data data/m2rag/mmqa/train_data.jsonl \
--mmqa_retrieve_data output/retrieval/mmqa/webqa_mmqa_train_retrieval_multi_5.jsonl \
--fact_verify_data data/m2rag/fact_verify/train_data.jsonl \
--fact_verify_retrieve_data output/retrieval/fact_verify/factify_fact_verify_train_retrieval_multi_5.jsonl \
--image_cap_data data/m2rag/image_cap/train_data.jsonl \
--image_cap_retrieve_data output/retrieval/image_cap/webqa_image_cap_train_retrieval_multi_5.jsonl \
--output_dir data/finetune \
--llm_type minicpmv \
--topk 5