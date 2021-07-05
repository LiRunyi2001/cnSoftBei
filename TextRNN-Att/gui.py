import PySimpleGUI as sg
import time
from importlib import import_module
import argparse
from utils import one_sentence_prep, build_iterator
from train_eval import predict_one_sentence

parser = argparse.ArgumentParser(description='TextRNN-Attention')
parser.add_argument('--model', default='TextRNN_Att', type=str, required=True,
                    help='choose a model: TextRNN_Att, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':

    sg.change_look_and_feel('LightBrown3')

    # Define the window's contents
    layout = [[sg.Text("请输入待测新闻", font='宋体')],
              [sg.Multiline(size=(20, 10), key='-INPUT-', font='宋体')],
              [sg.Text(size=(20, 1), key='-OUTPUT-', font='微软雅黑')],
              [sg.Button('预测', font='宋体'), sg.Button('退出', font='宋体')]]

    # Create the window
    window = sg.Window('新闻分类', layout)

    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break

        input = values['-INPUT-']
        with open("Toutiao/data/customer.txt", "w", encoding='utf-8') as f:
            f.write(input + '\n')

        # Output a message to the window
        label_map = {"1": "财经", "2": "房产", "3": "教育", "4": "科技", "5": "军事", "6": "汽车", "7": "体育", "8": "游戏", "9": "娱乐",
                     "10": "其他"}

        dataset = 'Toutiao'  # 数据集
        embedding = 'embedding_SougouNews.npz'

        if args.embedding == 'random':
            embedding = 'random'
        model_name = args.model  # TextRNN_Att, Transformer

        x = import_module('models.' + model_name)
        config = x.Config(dataset, embedding)

        # prepared the text
        vocab, prepared_text = one_sentence_prep(config, args.word)

        test_iter = build_iterator(prepared_text, config)

        # predict
        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)
        result = predict_one_sentence(config, model, test_iter)
        result = "".join(result)

        window['-OUTPUT-'].update(f"分类结果：{result}")

    # Finish up by removing from the screen
    window.close()