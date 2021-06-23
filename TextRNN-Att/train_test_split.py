from sklearn.model_selection import train_test_split

dic = {'财经': 0, '房产': 1, '教育': 2, '科技': 3, '军事': 4, '汽车': 5, '体育': 6, '游戏': 7, '娱乐': 8, '其他': 9}


def split(filename):
    with open(filename, encoding='utf-8') as f:
        lines = []
        rows = f.readlines()
        for row in rows:
            row = row.strip('\n')
            label = row[:2]
            label = dic[label]
            text = row[3:]
            row = text + '\t' + str(label)
            lines.append(row)

    f_train, test = train_test_split(lines, test_size=0.05, random_state=777)
    train, dev = train_test_split(f_train, test_size=0.05, random_state=777)

    with open("train.txt", "w+", encoding='utf-8') as f:
        for i in train:
            f.write(i + '\n')

    with open("dev.txt", "w+", encoding='utf-8') as f:
        for i in train:
            f.write(i + '\n')

    with open("test.txt", "w+", encoding='utf-8') as f:
        for i in train:
            f.write(i + '\n')


if __name__ == '__main__':
    split('toutiao.txt')
