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
# 사용자 정의 함수
# - 모델 관련 함수
# -------------------------------------
class RNN(nn.Module):  # RNN 모델 정의
    def __init__(self):
        super(RNN, self).__init__()
        # input size = 12
        self.rnn = nn.RNN(input_size=12, hidden_size=30, num_layers=2, batch_first=True)
        self.fc = nn.Linear(30, 1)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])
        return x


def denorms(x):  # 정규화된 값을 원래 값으로 변환
    return x * 43.89968 + 101.88185


def predict_df(data, start_date, pred_length):  # 모델을 이용한 예측
    # 1. form에서 받은 start_date로 index 찾기
    # 2. 해당 index부터 12개월씩 데이터를 모델에 넣어 pred_length만큼을 예측
    # 3. 예측 결과를 DataFrame으로 만들어 반환
    # 4. labels, data1, data2를 반환

    # date가 '2023-04-30' 이후이면 에러 출력
    if start_date > "2023-04-30":
        print("date error: 2023-04-30 이전으로만 입력해주세요 (학습 기간 : 12개월)")
        return
    else:
        start_idx = data[data["date"] == start_date].index.item()
        pred_total = data["norm"][start_idx : start_idx + 12].tolist()

        for i in range(pred_length):
            values = torch.tensor(pred_total[-12:]).unsqueeze(0).float()
            pred = model(values.unsqueeze(0))
            pred_total = np.append(pred_total, pred.item())
        pred_total_denorm = denorms(pred_total).tolist()
        pred_df = pd.DataFrame(
            {
                "date": pd.date_range(start_date, periods=pred_length, freq="M"),
                "value": pred_total_denorm[-pred_length:],
            }
        )
        return pred_df


# -------------------------------------
# 웹 페이지의 form 태그 내의 input 태그 입력값을 저장하는 인스턴스
# -------------------------------------
form = cgi.FieldStorage()  # input된 form 데이터를 받아옴


# -------------------------------------
# 클라이언트의 요청 데이터 추출
# - 입력 데이터 : start_date, pred_length
# - predict_df 함수를 이용하여 예측 결과 생성 -> DF
# - 예측 결과를 아래 HTML 템플릿에 삽입 : labels, data1, data2
# -------------------------------------
# form에서의 입력 데이터 추출
if "start_date" in form and "pred_length" in form:
    start_date = form.getvalue("start_date")
    pred_length = int(form.getvalue("pred_length"))

else:
    # default 값 설정
    start_date = "2023-04-30"
    pred_length = 6
# -------------------------------------
try:
    # 데이터 로딩
    data = pd.read_csv("../../DATA/SMP_201004_202403_norm.csv")

    model = RNN()
    model_path = "../models/noo_model_12m.pth"
    model.load_state_dict(torch.load(model_path))

    # predict_df
    predict_df = predict_df(data, start_date, pred_length)

    # HTML용 데이터 생성
    labels = json.dumps(predict_df["date"].dt.strftime("%Y-%m").tolist())
    data1 = json.dumps(data["value"].tolist())
    data2 = json.dumps(predict_df["value"].tolist())

except Exception as e:
    print("Error: ", e)
    labels = "['2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09']"
    data1 = "[100, 200, 300, 400, 500, 600]"
    data2 = "[150, 250, 350, 450, 550, 650]"


# -------------------------------------
# 요청에 대한 응답 HTML
# chart.js를 이용한 그래프 출력
# -------------------------------------
# HTML 템플릿

html_text = """
<!DOCTYPE html>
<html>
    <head>
        <title>예측 결과</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <canvas id="myChart"></canvas>
        <script>
            var ctx = document.getElementById('myChart').getContext('2d');
            var myChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: <!--LABELS-->,
                    datasets: [{
                        label: '실제값',
                        data: <!--DATA1-->,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }, {
                        label: '예측값',
                        data: <!--DATA2-->,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        </script>
    </body>
</html>
"""

# -------------------------------------
# 예측 결과 출력
# -------------------------------------
print("Content-Type: text/html; charset=utf-8;")
print()
# HTML Body
# - 위 템플릿에 데이터를 삽입(replace)하여 출력
print("<h1 style='text-align: center'>예측 결과</h1>")
print(
    html_text.replace("<!--LABELS-->", labels)
    .replace("<!--DATA1-->", data1)
    .replace("<!--DATA2-->", data2)
)
# 출력 테스트
# print("<p> print test </p>")
