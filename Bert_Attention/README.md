# Bert + Attention
## 简介

用Bert和Attention实现对中文的新闻文本进行分类。

## 数据集

头条数据集（UTF-8格式）。

标签: 财经、房产、教育、科技、军事、汽车、体育、游戏、娱乐、其他

## 环境

* `Tensorflow` == 1.14.0
* `Keras` == 2.3.1
* `bert4keras` == 0.8.4

## 文件说明

* `data_utils`：用于预训练语料的构建。
* `pretraining`：用于Bert的预训练。
* `train`：用于新闻文本分类模型的训练。
* `pred`：用于新闻文本分类模型的预测。

