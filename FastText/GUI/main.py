import PySimpleGUI as sg
import fasttext
# Define the window's contents
layout = [[sg.Text("请输入待测新闻", font='微软雅黑')],
          [sg.Multiline(size=(40, 20), key='-INPUT-', font='微软雅黑')],
          [sg.Text(size=(40, 1), key='-OUTPUT-', font='微软雅黑')],
          [sg.Button('预测', font='微软雅黑'), sg.Button('退出', font='微软雅黑')]]

# Create the window
window = sg.Window('新闻分类', layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == '退出':
        break
    # Output a message to the window
    label_map = {"1": "财经", "2": "房产", "3": "教育", "4": "科技", "5": "军事", "6": "汽车", "7": "体育", "8": "游戏", "9": "娱乐", "10": "其他"}
    model = fasttext.load_model("./model.bin")
    result = label_map[model.predict(values['-INPUT-'].strip())[0][0].split("__")[-1]]
    window['-OUTPUT-'].update(f"分类结果：{result}")

# Finish up by removing from the screen
window.close()