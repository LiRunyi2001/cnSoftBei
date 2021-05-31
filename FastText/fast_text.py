import fasttext
import jieba

from sklearn.model_selection import train_test_split


def load_stopwords():
    with open("./stopwords.txt") as file:
        stopwords = file.readlines()
        file.close()
    return [word.strip() for word in stopwords]


def write2txt(dt, path):
    with open(path, 'w') as f:
        for i in range(len(dt)):
            s = str(dt[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',') + '\n'
            f.write(s)


def preprocessing(path):
    label_map = {"财经": 1, "房产": 2, "教育": 3, "科技": 4, "军事": 5, "汽车": 6, "体育": 7, "游戏": 8, "娱乐": 9, "其他": 10}
    stop_words = load_stopwords()
    prep_data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            split_line = line.split(',')[:2]
            label, content = split_line
            exclude_sw = [word for word in jieba.cut(content) if word not in stop_words]
            content = ''.join([x.strip() for x in exclude_sw if x.strip() != ''])
            prep_data.append(f'__label__{label_map[label]} ' + content)
    train_data, test_data = train_test_split(prep_data, test_size=0.3, random_state=0)
    train_data_path, test_data_path = path[:-4] + "_train.txt", path[:-4] + "_test.txt"
    write2txt(train_data, train_data_path)
    write2txt(test_data, test_data_path)
    return train_data_path, test_data_path


if __name__ == "__main__":
    train_path, test_path = preprocessing('../toutiao.txt')
    model = fasttext.train_supervised(input=train_path, autotuneValidationFile=test_path, autotuneDuration=800)
    model.save_model('toutiao_model.bin')
    result = model.test(test_path)
    print(*result)
