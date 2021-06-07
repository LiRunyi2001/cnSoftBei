from sklearn.model_selection import train_test_split


def process_data(filename):
    with open(filename, encoding='utf-8') as f:
        lines = []
        rows = f.readlines()
        for row in rows:
            row = row.strip('\n')
            lines.append(row)
        train, test = train_test_split(lines, test_size=0.3, random_state=777)
    with open("toutiao_train.txt", "w+", encoding='utf-8') as f:
        for i in train:
            f.write(i + '\n')
    with open("toutiao_test.txt", "w+", encoding='utf-8') as f:
        for i in test:
            f.write(i + '\n')
    return train, test


if __name__ == '__main__':
    process_data('toutiao.txt')
