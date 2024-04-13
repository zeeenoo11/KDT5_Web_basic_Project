import cgi
import json
import sys
import codecs
from model import *

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def print_browser(output_count=1):
    filename = 'cgi-bin/index.html'
    with open(filename, 'r', encoding='utf-8') as f:
        file_contents = f.read()

        # test.csv를 읽은 후, 데이터를 JSON 형식으로 변환
        df = pd.read_csv("cgi-bin/test.csv")
        labels = list(df.index)
        data1 = list(df['0'].values)
        data2 = list(df['1'].values)
    
        chart_data = json.dumps({'labels': labels, 'data1': data1, 'data2': data2})
        file_contents_with_data = file_contents.replace('<!--CHART_DATA-->', chart_data)

        # HTML Header
        print("Content-Type: text/html; charset=utf-8;")
        print()
        # HTML Body
        print(file_contents_with_data)


form = cgi.FieldStorage()    
# output count, input count에 값이 있으면 실행
if 'output count' in form and 'input count' in form:
    # output count, input count 값을 가져옴
    output_count = int(form['output count'].value)
    input_count = int(form['input count'].value)
    print(output_count, input_count)
    df = pd.read_csv("../data/02_시간별 제주 전력수요량.csv", encoding="UTF-8" )
    df["sum"] = df.sum( axis = 1, numeric_only=True)
    scaler = MinMaxScaler()
    scale_cols = ['sum']
    df_scaled = scaler.fit_transform(df[scale_cols])
    df = pd.DataFrame(df_scaled, columns=scale_cols)
    dataset = DemandDataset(df)
    predict(input_count)
    
print_browser()



