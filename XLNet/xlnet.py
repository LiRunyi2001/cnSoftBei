import os
import csv
import torch
import transformers
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import adamw
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

transformers.logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-xlnet-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_score = 0
batch_size = 8
classes_list = []
train_epoch = 20

output_dir = './models/'
output_model_file = os.path.join(output_dir, 'model_file')
output_config_file = os.path.join(output_dir, 'config_file')


def process_data(filename):
    with open(filename, encoding='utf-8') as f:
        lines = []
        rows = f.readlines()
        for row in rows:
            row = row.strip('\n')
            lines.append(row)
        label_list = [line[0:2] for line in lines]
        global classes_list
        classes_list = list(set(label_list))  # non-repeated label list
        num_classes = len(classes_list)  # num of classes
        for i in range(len(label_list)):
            label_list[i] = classes_list.index(label_list[i])  # index of label

        sentence_list = [line[4:] for line in lines]
    return sentence_list, label_list, classes_list, num_classes


def convert(sentences, target):  # sentence_list, label_list
    input_ids, token_type_ids, attention_mask = [], [], []
    for i in range(len(sentences)):
        encoded_dict = tokenizer.encode_plus(
            sentences[i],  # input text
            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
            max_length=512,  # fill & trunc
            pad_to_max_length=True,
            return_tensors='pt',  # return pytorch tensors format
        )
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    input_ids = torch.LongTensor(input_ids)
    token_type_ids = torch.LongTensor(token_type_ids)
    attention_mask = torch.LongTensor(attention_mask)
    target = torch.LongTensor(target)

    return input_ids, token_type_ids, attention_mask, target


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()  # [3, 5, 8, 1, 2, ....]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def save(model):
    # save
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)


def eval(model, validation_dataloader):
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
    f1_macro, f1_weighted = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].cpu().numpy()
            # Validation accuracy
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            print('tmp_eval_accuracy=', tmp_eval_accuracy)
            nb_eval_steps += 1
            # F1_score weighted
            pred_flat_weighted = np.argmax(logits, axis=1).flatten()
            labels_flat_weighted = label_ids.flatten()
            tmp_f1_weighted = f1_score(labels_flat_weighted, pred_flat_weighted, average='weighted')
            f1_weighted += tmp_f1_weighted
            print('tmp_f1_weighted=', tmp_f1_weighted)
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("F1_score weighted: {}".format(f1_weighted / nb_eval_steps))
    global best_score
    if best_score < f1_weighted / nb_eval_steps:
        print('Above is the best result so far.')
        best_score = f1_weighted / nb_eval_steps
        save(model)


def train_eval(file='toutiao.txt'):
    sentence_list, label_list, _, num_classes = process_data(file)
    input_ids, token_type_ids, attention_mask, labels = convert(sentence_list, label_list)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=1,
                                                                          test_size=0.01)
    train_token, val_token, _, _ = train_test_split(token_type_ids, labels, random_state=1, test_size=0.01)
    train_mask, val_mask, _, _ = train_test_split(attention_mask, labels, random_state=1, test_size=0.01)

    train_data = Data.TensorDataset(train_inputs, train_token, train_mask, train_labels)
    train_dataloader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = Data.TensorDataset(val_inputs, val_token, val_mask, val_labels)
    validation_dataloader = Data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained('hfl/chinese-xlnet-base', num_labels=num_classes).to(
        device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=1e-3)

    for e in range(train_epoch):
        for i, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            loss = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3])[0]
            print('epoch{}\tloss={:.6f}'.format(e,loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if i % 1000 == 0 and i>0:
        eval(model, validation_dataloader)
        print('epoch{}/{}\tloss={:.6f}'.format(e, train_epoch, loss.item()))


# def pred(upload_file='toutiao_test.txt'):
#     # load model
#     model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
#
#     with open(upload_file) as f:
#         rows = [row for row in csv.reader(f)]
#         rows = np.array(rows[1:])
#         sentence_list = [text for idx, text in rows]
#
#     input_ids, token_type_ids, attention_mask, _ = convert(sentence_list, [1])  # whatever name_list and label_list
#     dataset = Data.TensorDataset(input_ids, token_type_ids, attention_mask)
#     loader = Data.DataLoader(dataset, 32, False)
#
#     pred_label = []
#     model.eval()
#     for i, batch in enumerate(loader):
#         batch = tuple(t.to(device) for t in batch)
#         with torch.no_grad():
#             logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
#             logits = logits.detach().cpu().numpy()
#             preds = np.argmax(logits, axis=1).flatten()
#             pred_label.extend(preds)
#
#     for i in range(len(pred_label)):
#         pred_label[i] = classes_list[pred_label[i]]
#
#     pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv('pred.csv')


if __name__ == '__main__':
    train_eval(file='toutiao.txt')
    # pred(upload_file='toutiao_test.txt')
