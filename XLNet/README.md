# Chinese-XLNet News

## Description

A method to classify news specifically in Chinese. 

We fine-tune the pre-trained [pretrained Chinese-XLNet model](https://github.com/ymcui/Chinese-XLNet) on Toutiao dataset. 

This method can predict the labels of one sentence at one time  or multiple sentences written in a .txt file.

F1_score = 

Validation accuracy = 

You may also interested in:

* Chinese XLNet: <https://github.com/ymcui/Chinese-XLNet>
* XLNet:  <https://github.com/zihangdai/xlnet>

## Requirements

* Python 3.7
* torch 1.4.0
* pip
* Huggingface-Transformers

## Quick start

* down load the model.

* Modify the value of `sentence` in `predict.py` to your desired sentence. Or, modify the value of `file_path` in `predict.py` .

* Run `predict.py`, and the label will be printed to the terminal. Or, the labels will be saved in `pred.csv` if a .txt file is uploaded.

  ```
  python predict.py
  ```

## Dataset

Toutiao Dataset.

Labels:  财经、房产、教育、科技、军事、汽车、体育、游戏、娱乐、其他

This dataset is encoded in UTF-8.