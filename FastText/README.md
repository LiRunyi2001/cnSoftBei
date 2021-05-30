# Fast Text
## Description
A very simple and fast method used to classify news.   
Precision and Recall: approximate 88% on the Toutiao Dataset
## Usage
**Only support querying one sentence at a time**
0. Install dependencies
   ```shell
   pip install requirements.txt
   ```
1. Modify `query_words` as the sentence you want to query.
2. Run `predict.py` and it will print the label according to `toutiao_model.bin`.
    ```shell
   python predict.py
   ```