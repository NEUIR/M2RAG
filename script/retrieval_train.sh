# The script is used to retrieve the multi-modal candidates for query in all tasks.

# ---------Task-1: image_cap----------
python ../src/retrieval.py \
--query_embed_path ../output/embed/image_cap/webqa_image_cap_query_images_train_embedding.pkl \
--cand_multi_embed_path ../output/embed/image_cap/webqa_cands_images_caps_embedding.pkl \
--data_path ../data/m2rag/image_cap/train_data.jsonl \
--dataset_name webqa \
--retrieval_modality multi \
--out_path ../output/retrieval \
--task image_cap \
--topN 5 \
--flag train

python ../src/get_retrieval_multi_data.py \
--cand_multi_path ../data/m2rag/image_cap/all_imgs_new.json \
--retrieval_path ../output/retrieval/image_cap/webqa_image_cap_query_train_visualbge_5_multi.trec \
--dataset_name webqa \
--retrieval_modality multi \
--out_path ../output/retrieval \
--task image_cap \
--topN 5 \
--flag train



# ---------Task-2: mmqa----------
python ../src/retrieval.py \
--query_embed_path ../output/embed/mmqa/webqa_mmqa_query_text_train_embedding.pkl \
--cand_image_embed_path ../output/embed/mmqa/webqa_cands_images_caps_embedding.pkl \
--cand_text_embed_path ../output/embed/mmqa/webqa_cands_text_docs_embedding.pkl \
--data_path ../data/m2rag/mmqa/train_data.jsonl \
--dataset_name webqa \
--retrieval_modality multi \
--out_path ../output/retrieval \
--task mmqa \
--topN 5 \
--flag train

python ../src/get_retrieval_multi_data.py \
--cand_path ../data/m2rag/mmqa/cand_text_docs.jsonl \
--cand_second_path ../data/raw_data/webqa/all_imgs.json \
--retrieval_path ../output/retrieval/mmqa/webqa_mmqa_query_train_visualbge_5_multi.trec \
--dataset_name webqa \
--retrieval_modality multi \
--out_path ../output/retrieval \
--task mmqa \
--topN 5 \
--flag train


# ---------Task-3: fact_verify----------
python ../src/retrieval.py \
--query_embed_path ../output/embed/fact_verify/factify_fact_verify_query_claim_images_train_embedding.pkl \
--cand_image_embed_path ../output/embed/fact_verify/factify_cands_images_embedding.pkl \
--cand_text_embed_path ../output/embed/fact_verify/factify_cands_text_embedding.pkl \
--data_path ../data/m2rag/fact_verify/train_data.jsonl \
--dataset_name factify \
--retrieval_modality multi \
--out_path ../output/retrieval \
--task fact_verify \
--topN 5 \
--flag train


python ../src/get_retrieval_multi_data.py \
--cand_path ../data/m2rag/fact_verify/cand_document.jsonl \
--cand_second_path ../data/m2rag/fact_verify/cand_images_map.jsonl \
--retrieval_path ../output/retrieval/fact_verify/factify_fact_verify_train_visualbge_5_multi.trec \
--dataset_name factify \
--retrieval_modality multi \
--out_path ../output/retrieval \
--task fact_verify \
--topN 5 \
--flag train
