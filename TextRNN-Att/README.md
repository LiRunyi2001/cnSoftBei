# TextRNN-Attention
采用BiLSTM+Attention模型，进行中文新闻文本分类。

## 环境
python 3.7  

pytorch 1.1  

tqdm  

sklearn  

tensorboardX

### 下载

* 未分割的头条数据集：[toutiao.txt](https://1drv.ms/t/s!AkxNwDoH9nwHgyqpC70xBTybYdS_?e=LWplcc)
* Toutiao文件夹下需要的数据集（直接复制到Toutiao下即可使用）：[Toutiao](https://1drv.ms/u/s!AkxNwDoH9nwHgyl5BZ4lQpiVtazm?e=qhJkmz)
* 模型下载：[TextRNN_Att](https://1drv.ms/u/s!AkxNwDoH9nwHgzJ7cJb3K9vufYJr?e=E54TDc)


### 使用自己的数据集
1. 按照[toutiao.txt](https://1drv.ms/t/s!AkxNwDoH9nwHgyqpC70xBTybYdS_?e=LWplcc)的格式调整自己的数据集格式。
2. 将 `train_test_split.py` 中 `split('toutiao.txt')` 中的文件换成自己的数据集。
3. 执行 `python train_test_split.py` 得到 `train.txt`，`dev.txt`, `test.txt`，可根据需要修改参数 `test_size`。
4. 把得到的数据移动到 `Toutiao` 下。


## 效果

模型|train_acc|val_acc|备注
--|--|--|--
 TextRNN_Att | 94.53%    | 93.31%  |BiLSTM+Attention

## 使用说明

* 模型训练

  ```go
  python run.py --model TextRNN_Att
  ```

  
