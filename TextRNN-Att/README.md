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
* 模型下载：[TextRNN_Att](https://1drv.ms/u/s!AkxNwDoH9nwHgzJ7cJb3K9vufYJr?e=gULMPU)


### 使用自己的数据集
1. 按照 [toutiao.txt](https://1drv.ms/t/s!AkxNwDoH9nwHgyqpC70xBTybYdS_?e=LWplcc) 的格式调整自己的数据集格式。
2. 将 `train_test_split.py` 中 `split('toutiao.txt')` 中的文件换成自己的数据集。
3. 执行 `python train_test_split.py` 得到 `train.txt`，`dev.txt`, `test.txt`，可根据需要修改参数 `test_size`。
4. 把得到的数据移动到 `Toutiao` 文件夹下。


## 效果

模型|precision|recall|f1_score
--|--|--|--
 TextRNN_Att | 96.37% | 96.41% |96.17%

## 使用说明

* 模型训练

  ```go
  python run.py --model TextRNN_Att
  ```


* 新闻测试

  * 下载预训练模型 [TextRNN_Att](https://1drv.ms/u/s!AkxNwDoH9nwHgzJ7cJb3K9vufYJr?e=gULMPU) 至 `Toutiao/saved_dict` 文件夹下。

  * 在 `data` 文件夹下的  `customer.txt` 中添加测试数据（可以是1条，也可以是多条），测试数据的格式需要与其中的示例数据保持一致。 然后执行下面的代码。

  `````
  python predict.py --model TextRNN_Att
  `````

  

