# The script is used to encode the query and multi-modal candidates.


# ---------Task-1: image_cap----------
# test_query
export CUDA_VISIBLE_DEVICES=0
python ../src/get_emb.py \
--model_weight ../pretrained_model/Visualized_base_en_v1.5.pth \
--out_path ../output/embed/ \
--image_path ../data/m2rag/image_cap/test_data.jsonl \
--dataset_name webqa \
--task image_cap \
--encode_image \
--is_query

# cands
export CUDA_VISIBLE_DEVICES=0
python ../src/get_emb.py \
--model_weight ../pretrained_model/Visualized_base_en_v1.5.pth \
--out_path ../output/embed/ \
--image_path ../data/m2rag/image_cap/all_imgs_new.json \
--img_linelist_path ../data/raw_data/webqa/imgs.lineidx.new \
--img_feat_path ../data/raw_data/webqa/imgs.tsv \
--dataset_name webqa \
--task image_cap \
--encode_image



# ---------Task-2: mmqa----------
# test_query
export CUDA_VISIBLE_DEVICES=1
python ../src/get_emb.py \
--model_weight ../pretrained_model/Visualized_base_en_v1.5.pth \
--out_path ../output/embed/ \
--image_path ../data/m2rag/mmqa/test_data.jsonl \
--dataset_name webqa \
--task mmqa \
--encode_text \
--is_query

# cands
export CUDA_VISIBLE_DEVICES=1
python ../src/get_emb.py \
--model_weight ../pretrained_model/Visualized_base_en_v1.5.pth \
--out_path ../output/embed/ \
--image_path ../data/raw_data/webqa/all_imgs.json \
--img_linelist_path ../data/raw_data/webqa/imgs.lineidx.new \
--img_feat_path ../data/raw_data/webqa/imgs.tsv \
--text_path ../data/m2rag/mmqa/cand_text_docs.jsonl \
--dataset_name webqa \
--task mmqa \
--encode_text \
--encode_image


# ---------Task-3: fact_verify----------
# test_query
export CUDA_VISIBLE_DEVICES=2
python ../src/get_emb.py \
--model_weight ../pretrained_model/Visualized_base_en_v1.5.pth \
--out_path ../output/embed/ \
--image_path ../data/m2rag/fact_verify/val_data.jsonl \
--dataset_name factify \
--task fact_verify \
--encode_image \
--is_query

# cands
export CUDA_VISIBLE_DEVICES=2
python ../src/get_emb.py \
--model_weight ../pretrained_model/Visualized_base_en_v1.5.pth \
--out_path ../output/embed/ \
--image_path ../data/m2rag/fact_verify/cand_images_map.jsonl \
--text_path ../data/m2rag/fact_verify/cand_document.jsonl \
--dataset_name factify \
--task fact_verify \
--encode_text \
--encode_image

# ---------Task-4: image_rerank----------
# test_query
export CUDA_VISIBLE_DEVICES=3
python ../src/get_emb.py \
--model_weight ../pretrained_model/Visualized_base_en_v1.5.pth \
--out_path ../output/embed/ \
--image_path ../data/m2rag/image_rerank/test_data.jsonl \
--dataset_name webqa \
--task image_rerank \
--encode_text \
--is_query

# cands
export CUDA_VISIBLE_DEVICES=3
python ../src/get_emb.py \
--model_weight ../pretrained_model/Visualized_base_en_v1.5.pth \
--out_path ../output/embed/ \
--image_path ../data/raw_data/webqa/all_imgs.json \
--img_linelist_path ../data/raw_data/webqa/imgs.lineidx.new \
--img_feat_path ../data/raw_data/webqa/imgs.tsv \
--dataset_name webqa \
--task image_rerank \
--encode_image