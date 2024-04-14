import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.utils.data import Dataset
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
    
if __name__=="__main__":
    df = pd.read_csv("../data/02_시간별 제주 전력수요량.csv", encoding="UTF-8" )
    df["sum"] = df.sum( axis = 1, numeric_only=True)
    scaler = MinMaxScaler()
    scale_cols = ['sum']
    df_scaled = scaler.fit_transform(df[scale_cols])
    df = pd.DataFrame(df_scaled, columns=scale_cols)
    dataset = DemandDataset(df)
    predict(30)

    



    