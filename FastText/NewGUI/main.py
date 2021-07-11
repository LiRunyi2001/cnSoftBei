import PySimpleGUI as sg
import pandas as pd
import fasttext
import re
sg.theme('TealMono')  # please make your windows colorful

layout = [[sg.Text('请输入数据集文件路径', font='微软雅黑')],
            [sg.Input(), sg.FileBrowse("浏览", font='微软雅黑')],
            [sg.OK(button_text = "确定", font='微软雅黑'), sg.Cancel("退出", font='微软雅黑')] ]

window = sg.Window('新闻分类', layout)
event, values = window.read()
if event == "确定":
    source_filename = values[0]
    sg.popup("数据处理中...", font='微软雅黑')
    dt = pd.read_excel(source_filename, sheet_name="类别")
    label_map = {"1": "财经", "2": "房产", "3": "教育", "4": "科技", "5": "军事", "6": "汽车", "7": "体育", "8": "游戏", "9": "娱乐", "10": "其他"}
    model = fasttext.load_model("./model.bin")
    str_strip = lambda str: re.sub(r"\s+", "", str)
    dt['channelName'] = dt.apply(lambda row: label_map[model.predict(str_strip(row['title'] + row['content']))[0][0].split("__")[-1]], axis=1)
    dt.to_excel(source_filename, sheet_name="类别")
    sg.popup("分类结果已生成", font='微软雅黑')