import cgi
import json
import sys
import codecs
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.utils.data import Dataset

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

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






WINDOW=4000
class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.RNN(input_size=WINDOW, hidden_size=30, num_layers=2, batch_first=True)
            self.fc = nn.Linear(30, 1)
        def forward(self, x):
            x, _status = self.rnn(x)
            x = self.fc(x[:, -1])
            return x

class DemandDataset(Dataset):
    def __init__(self, df):
        self.data = df
        self.x = self.data.iloc[:, -1].values
    def __len__(self):
        return len(self.data)-WINDOW
    def __getitem__(self, idx):
        return self.x[idx:idx+WINDOW], self.x[idx+WINDOW]

def predict(WINDOW, dataset):
    # 모델 예측
    
    model = torch.load(f"./model/model{WINDOW}.pth")
    model.eval()

    # 예측 데이터 생성
    pred = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        output = model(torch.tensor(x).unsqueeze(0).unsqueeze(1).float())
        pred.append(output.squeeze().detach().numpy())
        
    # 예측 데이터 합치기
    pred = np.array(pred).reshape(-1, 1)
    pred = np.concatenate([np.zeros((WINDOW, 1)), pred])
    # 예측 데이터 저장
    df["예측"] = pred
    data = list(df["예측"].values)
    sumdata = list(df["sum"].values)  
    print(len(sumdata), len(data))
    data = list(df["sum"].values)
    for i in range(10000):
        x = data[0+i:i+WINDOW]
        output = model(torch.tensor(x).unsqueeze(0).unsqueeze(1).float())
        if i>=5479-WINDOW:
            data.append(output.squeeze().detach().numpy().item())
    
    # 총 길이가 맞춰주기 
    for i in range(10000+WINDOW-5479):
        sumdata.append(0)
    print(len(sumdata), len(data))
    pd.DataFrame(zip(sumdata, data)).to_csv("test.csv")

    

form = cgi.FieldStorage()    
# output count, input count에 값이 있으면 실행
if 'output_count' in form and 'input_count' in form:
    # output count, input count 값을 가져옴
    output_count = int(form['output_count'].value)
    input_count = int(form['input_count'].value)
    print(output_count, input_count)
    df = pd.read_csv("./data/02_시간별 제주 전력수요량.csv", encoding="UTF-8" )
    df["sum"] = df.sum( axis = 1, numeric_only=True)
    scaler = MinMaxScaler()
    scale_cols = ['sum']
    df_scaled = scaler.fit_transform(df[scale_cols])
    df = pd.DataFrame(df_scaled, columns=scale_cols)
    dataset = DemandDataset(df)
    WINDOW=input_count
    predict(WINDOW = input_count, dataset=dataset)
    
    print_browser(output_count)

    


