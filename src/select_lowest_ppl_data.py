import json
import argparse
import os
import shutil
import concurrent.futures
import random


count = 0

def select_lowest_ppl(input_data, topk):
    task_dict = {}
    for start_idx in range(0, len(input_data), topk):
        one_task_list = input_data[start_idx: start_idx+ topk] 
        id = one_task_list[0][0]
        image_path = one_task_list[0][1]
        ppl = one_task_list[0][2]
        for item in one_task_list:
            assert item[0] == id
            if item[2] < ppl:
                image_path = item[1]
                ppl = item[2]
        task_dict[id] = image_path
    return task_dict


def read_file(input_file):
    with open(input_file) as fin:
        input_data = json.load(fin)
    return input_data

def write_file(output_data, output_file):
    with open(output_file, 'w') as fout:
        for id, image_path in output_data.items():
            fout.write(json.dumps({'id': id, 'image_path': image_path}) + '\n')

# move the image in top_list to output_dir
def restore_top_image(top_dict, output_dir):
    global count
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for id, image_path in top_dict.items():
        image_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_dir, image_name)
        if os.path.exists(output_image_path):
            print('repeated')
            image_names = image_name.split('.')
            image_name = image_names[0] + f'_copy{count}.' + image_names[1]
            count += 1
            output_image_path = os.path.join(output_dir, image_name)
        shutil.copy(image_path, output_image_path)
    print(f"All Top Image saved in {output_dir}.  Repeated count: {count}")

def rewrite_to_restore_path(top_dict, output_dir):
    for id, image_path in top_dict.items():
        image_name = os.path.basename(image_path)
        top_dict[id] = os.path.join(output_dir, image_name)
    return top_dict



if __name__ == '__main__':
    topk = 5

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="../output/image_rerank/test_qwen2vl_5_ppl.json")
    parser.add_argument('--output_file', type=str, default='../output/image_rerank/test_qwen2vl_5_ppl_selected.jsonl')
    parser.add_argument('--output_dir', type=str, default='../output/image_rerank/image_rerank_retrieve_qwen2vl_ppl_top')
    parser.add_argument('--restore_image', action='store_true', default=True)
    parser.add_argument('--topk', type=int, default=topk)

    args = parser.parse_args()
    input_data = read_file(args.input_file)
    output_data = select_lowest_ppl(input_data,args.topk)
    print("OUTPUT DATA LEN: ", len(output_data))

    if args.restore_image and args.output_dir != None:
        restore_top_image(output_data, args.output_dir)
        output_data = rewrite_to_restore_path(output_data, args.output_dir)

    write_file(output_data, args.output_file)
    print("Finished")

