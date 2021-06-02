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
1. Download the pretrained model from [here](https://1drv.ms/u/s!AmRrl2CAWm_2hNhj7Mu-V2Ku15dQKg?e=zxg1cA) and move it to the root directory.
2. Modify the value of `query_words` as the sentence you want to query in `predict.py`.
3. Run `predict.py` and it will print the label according to `toutiao_model.bin`.
    ```shell
   python -u predict.py
   ```
