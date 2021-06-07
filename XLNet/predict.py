import os
import csv
import torch
import transformers
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
import torch.utils.data as Data
from xlnet import convert, eval

transformers.logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dir = './models/'
output_model_file = os.path.join(output_dir, 'WEIGHTS_NAME')
output_config_file = os.path.join(output_dir, 'CONFIG_NAME')

batch_size = 32


def pred_one(sentence=None):
    # load model
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)

    if sentence == None:
        raise ValueError("The length of the sentence should be larger than 0.")

    sentence = np.array(sentence).tolist()

    input_ids, token_type_ids, attention_mask, _ = convert(sentence, [1])  # whatever name_list and label_list

    pred_label = []
    model.eval()

    with torch.no_grad():
        logits = model(sentence, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        pred_label.extend(preds)

    print('The predit label=', pred_label)


def pred_file(upload_file):
    # load model
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)

    with open(upload_file, encoding='utf-8') as f:
        lines = []
        rows = f.readlines()
        for row in rows:
            row = row.strip('\n')
            lines.append(row)
        sentence_list = [line for line in lines]

    input_ids, token_type_ids, attention_mask, _ = convert(sentence_list, [1])  # whatever name_list and label_list
    dataset = Data.TensorDataset(input_ids, token_type_ids, attention_mask)
    loader = Data.DataLoader(dataset, batch_size, False)

    pred_label = []
    model.eval()
    for i, batch in enumerate(loader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            pred_label.extend(preds)

    pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv('pred.csv')


if __name__ == '__main__':
    # if you want to predit the class of a sentence.
    sentence = 'hello.'
    pred_one(sentence)

    # if you want to predit the class of multiple sentences.
    file_path = ''
    pred_file(file_path)
