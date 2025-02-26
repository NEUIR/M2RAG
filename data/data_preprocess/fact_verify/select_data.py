import json
import random
import os
random.seed(2025)
def get_data(path):
    data=[]
    Support_Text=0
    Insufficient_Text=0
    Support_Multimodal=0
    Insufficient_Multimodal=0
    Refute=0
    
    with open(path,'r') as fin:
        for line in fin:
            if len(data)>=3000:
                break
            item=json.loads(line.strip())
            if len(item['document'].split())>2048:
                continue
            if item['category']=='Support_Text' and Support_Text<600:
                item['category']=='Support'
                data.append(item)
                Support_Text+=1
            elif item['category']=='Insufficient_Text' and Insufficient_Text<600:
                item['category']=='Insufficient'
                data.append(item)
                Insufficient_Text+=1
            elif item['category']=='Support_Multimodal' and Support_Multimodal<600:
                item['category']=='Support'
                data.append(item)
                Support_Multimodal+=1
            elif item['category']=='Insufficient_Multimodal' and Insufficient_Multimodal<600:
                item['category']=='Insufficient'
                data.append(item)
                Insufficient_Multimodal+=1
            elif item['category']=='Refute' and Refute<600:
                data.append(item)
                Refute+=1
            else:
                continue
    return data

def change_data(data):
    random.shuffle(data)
    for new_id, item in enumerate(data):
        item['id'] = f'val_{new_id}'
    return data

def write_data(data,outpath):
    with open(outpath, 'w') as fout:
        for item in data:
            json.dump(item, fout) 
            fout.write('\n')
    print(f"Data has been written to {outpath}")
 

if __name__=='__main__':
    test_data_path='../raw_data/factify/factify_val.jsonl'
    output_path='../m2rag/fact_verify/'
    os.makedirs(output_path, exist_ok=True)
    
    test_data=get_data(test_data_path)
    test_data=change_data(test_data)
    test_data_output_path=os.path.join(output_path,'val_data.jsonl')
    write_data(test_data,test_data_output_path)
    
    train_data_path='../raw_data/factify/factify_train.jsonl'
    train_data=get_data(train_data_path)
    train_data=change_data(train_data)
    train_data_output_path=os.path.join(output_path,'train_data.jsonl')
    write_data(train_data,train_data_output_path)
    
            
                
            