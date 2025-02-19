
# data process for webqa
# step-1
python ../src/get_similarirt.py \
--out_path ../data/raw_data/webqa \
--model_path ../pretrained_model/Visualized_base_en_v1.5.pth \
--cap_path ../data/raw_data/WebQA/all_imgs.json \
--img_feat_path ../data/raw_data/WebQA/imgs.tsv \
--img_linelist_path ../data/raw_data/WebQA/imgs.lineidx.new \
