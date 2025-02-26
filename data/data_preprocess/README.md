# The Data Processing Steps for $M^2RAG$

## 🎃 $M^2RAG$
$M^2RAG$ is built on **WebQA** and **Factify** datasets. 
* **WebQA** is a multi-hop, multi-modal, open-domain question answering benchmark. The dataset contains images and passage snippets crawled from the general Web and Wikipedia. 
* **Factify** is a multi-modal fact verification dataset. It is notably the largest multimodal fact verification public dataset consisting of 50K data points, covering news from India and the US. 

## 🎃 Quick Start
**🌵 1. Download WebQA and Factify datasets.**

* Please download these two datasets and put them in ```data/raw_data```.
* Download link for WebQA: [WebQA](https://thunlp.oss-cn-qingdao.aliyuncs.com/UniVLDR/data.zip) (❗️Note: For the ```imgs.tsv```, you need to download the data from [this link](https://drive.google.com/drive/folders/1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ?usp=sharing) and run ```7z x imgs.7z.001```).
* Get datasets for Factify: [Factify](https://competitions.codalab.org/competitions/35153#participate-get-data).

**🌵 2. Image Caption task.**
* Step-1: Preprocess the data to obtain similarity scores between images and captions, and filter out pairs with higher similarity scores.
```bash 
python image_cap/get_similarity.py
```

* Step-2: Split the dataset and build the retrieval corpus.
```bash 
python image_cap/select_data.py
```
**🌵 3. Multi-Modal Question Answering task.**
* Select an equal number of image-based and text-based questions to construct the training and test sets, and build the retrieval corpus.
```bash 
python mmqa/select_data.py
```

**🌵 4. Multi-Modal Fact Verification task.**
* Step-1: First, download the original Factify dataset. After downloading, place the train.csv and val.csv files in the following directory: ```../raw_data/factify/``` \
Next, run the ```download_images.py``` to download the images and filter out the data for which images cannot be downloaded. The result files are: ```../raw_data/factify/factify_train.jsonl``` and ```../raw_data/factify/factify_val.jsonl```. They will be used to Step-2: select data.
```bash
python fact_verify/download_images.py
```

* Step-2: Select an equal number of data from the five class labels to construct the training and test sets.
```bash 
python fact_verify/select_data.py
```

* Step-3: Build the retrieval corpus.
```bash 
python fact_verify/get_cands.py
```

**🌵 5. Image Reranking task.**
* Build the test set and retrieve the corpus.
```bash 
python image_rerank/select_data.py
```
**🌵 6. Images of WebQA**
* Finally, please move the ```imgs.lineidx.new``` and ```imgs.tsv``` to ```data/m2rag```

```bash 
mv ../raw_data/WebQA/imgs.lineidx.new ../m2rag
mv ../raw_data/WebQA/imgs.tsv ../m2rag
```

## 🎃 Contact
🎉 Congratulations on successfully building the dataset. 

📫 If you have any questions about the dataset or the processing steps, please feel free to contact us. 

```bash
zhuxingsheng@stumail.neu.edu.cn     zhoutianshuo@stumail.neu.edu.cn 
```