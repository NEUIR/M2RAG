import pandas as pd
import json
import os
from tqdm import tqdm
import shutil

def save_images(path,image):
    with open(path, 'wb') as f:
        f.write(image)
        print(f"Image saved in {path}")
        
def copy_image_to_target(target_image,original_image):
    
    shutil.copy(original_image, target_image)
    


def get_data(path1, path2,output_path):
    jsonl_files=[path1, path2]
    document_texts=[]
    document_images=[]
    for file_name in jsonl_files:
        with open(file_name,'r') as fin:
            for line in fin:
                item=json.loads(line.strip())
                document_texts.append(item['document'])
                document_images.append(item['document_image'])
                
    # save text
    documents_jsonl_path=os.path.join(output_path,'cand_document.jsonl')            
    with open(documents_jsonl_path, mode='w') as fout:
        for text_id, text in tqdm(enumerate(document_texts),desc="Processing texts"):
            json.dump({"id": f'text_{text_id}', "text": text}, fout) 
            fout.write('\n')
    print(f'The cands of document are saved in: {documents_jsonl_path}')
    
    # save images
    image_map = []
    id=0
    os.makedirs(os.path.join(output_path, 'cand_images'), exist_ok=True)
    
    for image in tqdm(document_images, desc="Processing Images"):
        image_type = os.path.splitext(image)[1].lstrip('.')
        image_path=os.path.join(output_path,'cand_images/image_{}.{}'.format(id,image_type))
        try:
            shutil.copy(image, image_path)
            image_map.append({'id':f'image_{id}','image_path':image_path})
            id+=1
            print(f'Successfully saved the image in {image_path}!！')
        except Exception as e:
            print(f'Save image {image} error at {image_path}! Error: {e}')
    
    #save image_map data
    image_map_path = os.path.join(output_path, "cand_images_map.jsonl")
    with open(image_map_path, mode='w') as fout:
        for item in tqdm(image_map, desc="Processing Images_map"):
            json.dump(item, fout) 
            fout.write('\n')
        print(f'The cands of image_map is saved in: {image_map_path}')
        

if __name__=='__main__':
    data_path_train='../raw_data/factify/factify_train.jsonl'
    data_path_dev='../raw_data/factify/factify_val.jsonl'
    output_path='../m2rag/fact_verify/'
    os.makedirs(output_path, exist_ok=True)
    get_data(data_path_train, data_path_dev,output_path)