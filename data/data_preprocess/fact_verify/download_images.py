import concurrent.futures
import os
import io
import time
import requests
from PIL import Image
import pandas as pd
import zipfile
import json
from concurrent.futures import ThreadPoolExecutor
import concurrent
from tqdm import tqdm

headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}


def download_images(n, row, claim_image_path, doc_image_path):
    success_download_list = []
    try:
        # download claim image
        response = requests.get(row['claim_image'], headers=headers, timeout=10)
        response.raise_for_status()
        claim_img = Image.open(io.BytesIO(response.content))
        claim_img = claim_img.convert('RGB')
        claim_image_output_path = os.path.join(claim_image_path, str(n) + '.jpg')
        claim_img.save(claim_image_output_path)

        # download document image
        response = requests.get(row['document_image'], headers=headers, timeout=10)
        response.raise_for_status()
        doc_img = Image.open(io.BytesIO(response.content))
        doc_img = doc_img.convert('RGB')
        doc_image_output_path = os.path.join(doc_image_path, str(n) + '.jpg')
        doc_img.save(doc_image_output_path)

        json_format_data = {
            'id':str(n),
            'claim':row['claim'],
            'claim_image':claim_image_output_path,
            'document':row['document'],
            'document_image':doc_image_output_path,
            'category':row['Category']
        }
        success_download_list.append(json_format_data)
        return success_download_list
    except Exception as e:
        print(f"Error downloading image of the {n}.th data")
        return []


def main():
    original_factify_train_csv_path='../raw_data/factify/train.csv'
    original_factify_val_csv_path='../raw_data/factify/val.csv'
    jsonl_output_path='../raw_data/factify/'

    train_claim_image_path = './train_images/'
    train_doc_image_path = './train_document_images/'
    val_claim_image_path = './val_images/'
    val_doc_image_path = './val_document_images/'

    os.makedirs(train_claim_image_path, exist_ok=True)
    os.makedirs(train_doc_image_path, exist_ok=True)
    os.makedirs(val_claim_image_path, exist_ok=True)
    os.makedirs(val_doc_image_path, exist_ok=True)

    train_df = pd.read_csv(original_factify_train_csv_path, index_col="Id")
    val_df = pd.read_csv(original_factify_val_csv_path, index_col="Id")

    all_train_data = []
    all_val_data = []

    with ThreadPoolExecutor() as executor:
        future_results = [executor.submit(download_images, n, row, train_claim_image_path, train_doc_image_path) for n, row in train_df.iterrows()]
        for future in concurrent.futures.as_completed(future_results):
            all_train_data.extend(future.result())

    with ThreadPoolExecutor() as executor:
        future_results = [executor.submit(download_images, n, row, val_claim_image_path, val_doc_image_path) for n, row in val_df.iterrows()]
        for future in concurrent.futures.as_completed(future_results):
            all_val_data.extend(future.result())
    
    with open(os.path.join(jsonl_output_path, 'factify_train.jsonl'), 'w') as fout:
        for item in all_train_data:
            fout.write(json.dumps(item) + '\n')

    with open(os.path.join(jsonl_output_path, 'factify_val.jsonl'), 'w') as fout:
        for item in all_val_data:
            fout.write(json.dumps(item) + '\n')


if __name__=='__main__':
    main()