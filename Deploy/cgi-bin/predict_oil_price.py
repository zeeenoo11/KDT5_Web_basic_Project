import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import cgi, sys, codecs, json

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


class OilPriceDataset(Dataset):
    def __init__(self, data, min_data=None, max_data=None, step=365):
        data = data if isinstance(data, np.ndarray) else data.values
        self.min_data = np.min(data) if min_data is None else min_data
        self.max_data = np.max(data) if max_data is None else max_data
        self.data = (data - self.min_data) / (self.max_data - self.min_data)
        self.data = torch.FloatTensor(self.data)
        self.step = step

    def __len__(self):
        return len(self.data) - self.step

    def __getitem__(self, i):
        data = self.data[i : i + self.step]
        label = self.data[i + self.step].squeeze()
        return data, label


class OilPriceModel(nn.Module):
    def __init__(self, hidden_size, num_layers, step):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(in_features=hidden_size * step, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.flatten(x)


def predict_price(select_date="2030-12-31"):
    filename = "./oil_data/oil_price.csv"
    priceDF = pd.read_csv(filename, encoding="utf-8", parse_dates=["date"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = 1825
    min_data = np.min(priceDF["price"].values)
    max_data = np.max(priceDF["price"].values)
    dataset = OilPriceDataset(
        priceDF[["price"]], min_data=min_data, max_data=max_data, step=step
    )

    start_date = priceDF["date"].max() + pd.DateOffset(days=1)
    end_date = pd.to_datetime(select_date)
    full_date_range = pd.date_range(start=start_date, end=end_date)
    pred_days = len(full_date_range)

    preds = []
    pred_model = torch.load(f"oil_price_model_{step}_cuda.pth", map_location=device)
    pred_model.eval()
    start = dataset[len(dataset.data[step:]) - 1][0]
    with torch.no_grad():
        for i in range(pred_days):
            pred = pred_model(start.unsqueeze(0).to(device))
            start = torch.cat((start[1:].to(device), pred.unsqueeze(0)))
            preds.append(pred.item())

    real_preds = [
        x * (float(max_data) - float(min_data)) + float(min_data) for x in preds
    ]

    pred_priceDF = pd.DataFrame({"date": full_date_range, "price": real_preds})
    pred_priceDF["price"] = pred_priceDF["price"].round(3)

    pred_priceDF.to_csv(f"./oil_data/pred_oil_price.csv", index=False, encoding="utf-8")


def print_browser(select_date="2030-12-31"):
    output_count = (pd.to_datetime(select_date) - pd.to_datetime("2024-04-12")).days
    filename = "./html/oil_price.html"
    with open(filename, "r", encoding="utf-8") as f:
        file_contents = f.read()

        # test.csv를 읽은 후, 데이터를 JSON 형식으로 변환
        oil_priceDF = pd.read_csv("./oil_data/oil_price.csv", encoding="utf-8")
        oil_pred_priceDF = pd.read_csv(
            f"./oil_data/pred_oil_price.csv", encoding="utf-8"
        )
        labels = (
            list(oil_priceDF["date"]) + list(oil_pred_priceDF["date"])[:output_count]
        )
        data1 = (
            list(oil_priceDF["price"]) + list(oil_pred_priceDF["price"])[:output_count]
        )
        data2 = list(oil_priceDF["price"])

        chart_data = json.dumps({"labels": labels, "data1": data1, "data2": data2})
        # 2030년 12월 31일의 예측 유가 : 99.999$/배럴
        chart_date = pd.to_datetime(select_date).strftime("%Y년 %m월 %d일") + '의 예측 유가 : ' + str(data1[-1]) + '$/배럴'
        file_contents_with_data = (
            file_contents.replace("REPLACE_DATA", chart_data)
            .replace("예측을 원하는 날짜를 선택하세요", chart_date)
        )

        # HTML Header
        print("Content-Type: text/html; charset=utf-8;")
        print()
        # HTML Body
        print(file_contents_with_data)


form = cgi.FieldStorage()
# output count, input count에 값이 있으면 실행
if "select_date" in form:
    select_date = str(form["select_date"].value)
    print(select_date)
    if "re_predict" in form:
        predict_price(select_date)

    print_browser(select_date)
