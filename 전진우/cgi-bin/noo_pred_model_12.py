# 모듈 로딩
import cgi, sys, codecs, os, cgitb, datetime, time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json


cgitb.enable()  # for debugging

# 한글 출력을 위한 인코딩 선언
# sys.stdout.reconfigure(encoding='utf-8')
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# -------------------------------------
# 웹 페이지의 form 태그 내의 input 태그 입력값을 저장하는 인스턴스
# -------------------------------------
form = cgi.FieldStorage()  # input된 form 데이터를 받아옴

# -------------------------------------
# 사용자 정의 함수
# -------------------------------------

def print_browser(output_count=1):
    filename = 'cgi-bin/index.html'
    with open(filename, 'r', encoding='utf-8') as f:
        file_contents = f.read()

        # test.csv를 읽은 후, 데이터를 JSON 형식으로 변환
        df = pd.read_csv("test.csv")
        labels = list(df.index)[:output_count]
        data1 = list(df['1'].values)[:output_count]
        data2 = list(df['0'].values)[:output_count]

        chart_data = json.dumps({'labels': labels, 'data1': data1, 'data2': data2})
        file_contents_with_data = file_contents.replace('<!--CHART_DATA-->', chart_data)

        # HTML Header
        print("Content-Type: text/html; charset=utf-8;")
        print()
        # HTML Body
        print(file_contents_with_data)

# -------------------------------------
# 모델 관련 함수
# -------------------------------------
class RNN(nn.Module):   # RNN 모델 정의
    def __init__(self):
        super(RNN, self).__init__()
        # input size = 12
        self.rnn = nn.RNN(input_size=12, hidden_size=30, num_layers=2, batch_first=True)
        self.fc = nn.Linear(30, 1)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])
        return x

def denorms(x):     # 정규화된 값을 원래 값으로 변환
    return x * 43.89968 + 101.88185

def predict_df(model_path, data, start_date, pred_length):  # 모델을 이용한 예측
    model = RNN()
    model.load_state_dict(torch.load(model_path))
    
    # date가 '2023-04-30' 이후이면 에러 출력
    if start_date > '2023-04-30':
        print('date error: 2023-04-30 이전으로만 입력해주세요 (학습 기간 : 12개월)')
        return
    else:
        start_idx = data[data['date'] == start_date].index.item()
        pred_total = data['norm'][start_idx: start_idx+12].tolist()
        
        for i in range(pred_length):
            values = torch.tensor(pred_total[-12:]).unsqueeze(0).float()
            pred = model(values.unsqueeze(0))
            pred_total = np.append(pred_total, pred.item())
        pred_total_denorm = denorms(pred_total).tolist()
        pred_df = pd.DataFrame({'date': pd.date_range(start_date, periods=pred_length, freq='M'), 'value': pred_total_denorm[-pred_length:]})
        return pred_df


# -------------------------------------
# 클라이언트의 요청 데이터 추출
# - 입력 데이터 : start_date, pred_length
# -------------------------------------
# 시간(초단위까지)을 이름에 붙임
# img_file = form.getvalue("img_file")
# filename = form.getvalue("filename")
# if img_file:
#     img_file = img_file + "_" + str(int(time.time())) + ".png"
#     with open(f"./image/{img_file}", "wb") as f:
#         f.write(form.getvalue("img_file"))
# else:
#     img_file = "None"
# if filename:
#     msg = filename
# else:
#     msg = "None"
# -------------------------------------
# if "img_file" in form and "filename" in form:
#     fileitem = form["img_file"]  # form.getvalue('img_file')로도 가능
#     img_file = fileitem.filename  # form.getvalue('message')로도 가능
    
#     suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     # 파일 저장
#     save_path = f"./image/{suffix}_{img_file}"
#     with open(save_path, "wb") as f:  # wb : write binary; byte로 쓰기
#         f.write(fileitem.file.read())  # filename으로 파일 저장

#     # 파일 경로 호출이 다름; 위는 .html, 아래는 .py
#     img_path = f"../image/{suffix}_{img_file}"  # 해당 파일 경로 저장
#     msg = form.getvalue("filename")
# else:
#     img_path = "None"
#     msg = "None"
# -------------------------------------
if "start_date" in form and "pred_length" in form:
    start_date = form.getvalue("start_date")
    pred_length = int(form.getvalue("pred_length"))
    
else:
    start_date = "2023-04-30"
    pred_length = 6


# -------------------------------------
# 요청에 대한 응답 HTML
# chart.js를 이용한 그래프 출력
# -------------------------------------
print("Content-Type: text/html; charset=utf-8")  # HTML로 출력; 이 문장이 header
print()
print("<TITLE>script output</TITLE>")
print("<H1>script output</H1>")


