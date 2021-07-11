import time
from importlib import import_module
import argparse
from utils import one_sentence_prep, build_iterator
from train_eval import predict_one_sentence
import pandas as pd

parser = argparse.ArgumentParser(description='TextRNN-Attention')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextRNN_Att, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'Toutiao'  # dataset
    embedding = 'embedding_SougouNews.npz'

    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # TextRNN_Att or Transformer
    from utils import get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    start_time = time.time()
    # prepared the text
    vocab, prepared_text = one_sentence_prep(config, args.word)

    test_iter = build_iterator(prepared_text, config)

    # predict
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)  # load the model
    res = predict_one_sentence(config, model, test_iter)
    print(len(res))  # show the result

    # save to a file.
    df = pd.DataFrame(res)
    df['channelName'] = res
    df.to_csv(r'./res.csv', columns=['channelName'], index=False, sep=',')

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
