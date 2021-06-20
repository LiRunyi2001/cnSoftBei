from sklearn.model_selection import train_test_split


def split(filename):
    with open(filename, encoding='utf-8') as f:
        lines = []
        rows = f.readlines()
        for row in rows:
            if row:
                row = row.strip('\n')
                lines.append(row)
        train, test = train_test_split(lines, test_size=0.01, random_state=777)
    with open("data/toutiao_train.csv", "w+", encoding='utf-8') as f:
        f.write('label' + ',' + 'text' + '\n')
        for i in train:
            f.write(i + '\n')
    with open("data/toutiao_test.csv", "w+", encoding='utf-8') as f:
        f.write('label' + ',' + 'text' + '\n')
        for i in test:
            f.write(i + '\n')
    return train, test

# def process_data(filename):
#     with open(filename, encoding='utf-8') as f:
#         lines = []
#         rows = f.readlines()
#         rows = [row.strip() for row in rows if row.strip() != '']
#         # print(rows)
#         for row in rows:
#             row = row.strip('\n')
#             lines.append(row)
#
#     with open("data/toutiao.csv", "w+", newline='', encoding='utf-8') as f:
#         for i in lines:
#             f.write(i + '\n')


if __name__ == '__main__':
    # process_data('toutiao.csv')
    split('toutiao.csv')
