# Copy data form image captioning task and WebQA.
# You can also direct use the test data form image captioning task for downstream tasks.
import shutil
import os
source_test = '../m2rag/image_cap/test_data.jsonl'
source_cand = "../raw_data/WebQA/all_imgs.json"
destination_path = '../m2rag/image_rerank'
os.makedirs(destination_path, exist_ok=True)

shutil.copy(source_test, destination_path)
shutil.copy(source_cand, destination_path)
    