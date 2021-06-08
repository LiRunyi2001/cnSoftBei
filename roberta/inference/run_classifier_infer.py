"""
  This script provides an exmaple to wrap UER-py for classification inference.
"""
import sys
import os
import torch
import argparse
import collections
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import infer_opts
from finetune.run_classifier import Classifier


def batch_loader(batch_size, src, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, seg_batch


# def read_dataset(args, path):
#     dataset, columns = [], {}
#     with open(path, mode="r", encoding="utf-8") as f:
#         for line_id, line in enumerate(f):
#             if line_id == 0:
#                 line = line.strip().split("\t")
#                 for i, column_name in enumerate(line):
#                     columns[column_name] = i
#                 continue
#             line = line.strip().split("\t")
#             if "text_b" not in columns:  # Sentence classification.
#                 # text_a = line[columns["text_a"]]
#                 # here the text_a indicates the user input
#                 text_a = "中国空军歼-20战机首次开展海上方向实战化训练, 歼-20战机,空天一体,申进科,空军,歼-20,中国空军"
#                 src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
#                 seg = [1] * len(src)
#             else:  # Sentence pair classification.
#                 text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
#                 src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
#                 src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
#                 src = src_a + src_b
#                 seg = [1] * len(src_a) + [2] * len(src_b)
#
#             if len(src) > args.seq_length:
#                 src = src[: args.seq_length]
#                 seg = seg[: args.seq_length]
#             while len(src) < args.seq_length:
#                 src.append(0)
#                 seg.append(0)
#             dataset.append((src, seg))
#
#     return dataset


def read_dataset(args, text_a):
    dataset, columns = [], {}
    # here the text_a indicates the user input
    # text_a = "中国空军歼-20战机首次开展海上方向实战化训练, 歼-20战机,空天一体,申进科,空军,歼-20,中国空军"
    src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
    seg = [1] * len(src)


    if len(src) > args.seq_length:
        src = src[: args.seq_length]
        seg = seg[: args.seq_length]
    while len(src) < args.seq_length:
        src.append(0)
        seg.append(0)
    dataset.append((src, seg))

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument("--load_model_path", default="models/model_1_epochs.bin", type=str,
    #                     help="Path of the input model.")
    # parser.add_argument("--vocab_path", default="models/google_zh_vocab.txt", type=str,
    #                     help="Path of the vocabulary file.")
    # parser.add_argument("--spm_model_path", default=None, type=str,
    #                     help="Path of the sentence piece model.")
    # parser.add_argument("--test_path", default="datasets/toutiao/test_nolabel.tsv", type=str, required=True,
    #                     help="Path of the testset.")
    # parser.add_argument("--prediction_path", type=str, required=True,
    #                     help="Path of the prediction file.")
    # parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
    #                     help="Path of the config file.")

    infer_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # 需要
    parser.add_argument("--labels_num", type=int,required=True,default=10,
                        help="Number of prediction labels.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    model = Classifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    text_input=input("请输入新闻文字\n")
    dataset = read_dataset(args, text_input)
    # print("dataset",dataset)
    # dataset = "勇士队主教练史蒂夫.科尔当年在公牛队时的三分球和库里比拼，谁更准？"

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("label")
        if args.output_logits:
            f.write("\t" + "logits")
        if args.output_prob:
            f.write("\t" + "prob")
        f.write("\n")
        for i, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)
            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)
            
            pred = torch.argmax(logits, dim=1)

            pred = pred.cpu().numpy().tolist()
            prob = nn.Softmax(dim=1)(logits)
            logits = logits.cpu().numpy().tolist()
            prob = prob.cpu().numpy().tolist()
            # print(pred[0])
            label=str(pred[0])
            if(label=='0'):
                print("财经")
            elif(label=='1'):
                print("房产")
            elif (label == '2'):
                print("教育")
            elif (label == '3'):
                print("科技")
            elif (label == '4'):
                print("军事")
            elif (label == '5'):
                print("汽车")
            elif (label == '6'):
                print("体育")
            elif (label == '7'):
                print("游戏")
            elif (label == '8'):
                print("娱乐")
            elif (label == '9'):
                print("其他")

            # print(logits)

            
            for j in range(len(pred)):
                f.write(str(pred[j]))
                if args.output_logits:
                    f.write("\t" + " ".join([str(v) for v in logits[j]]))
                if args.output_prob:
                    f.write("\t" + " ".join([str(v) for v in prob[j]]))
                f.write("\n")


if __name__ == "__main__":
    main()
