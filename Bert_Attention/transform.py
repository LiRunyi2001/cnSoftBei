import csv
import pandas as pd


def mymap(str) -> int:
    dict = {'财经': 0, "房产": 1, "教育": 2, "科技": 3, "军事": 4, "汽车": 5, "体育": 6, "游戏": 7, "娱乐": 8, "其他": 9}
    return dict[str]

reader = open('toutiao.txt', encoding='utf-8')
list_data = reader.readlines()
columns = ['label', 'text']
list = []

for i in list_data:
    i = i.strip('\r\n\n')
    label = mymap(str(i[0:2]))
    list.append([label, i[4:]])
print(list)
with open("toutiao.csv", "w", encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # 先写入columns_name
    writer.writerow(columns)
    # 写入多行用writerows
    writer.writerows(list)

# df = pd.read_csv('toutiao.csv')
# print(df.dropna())
