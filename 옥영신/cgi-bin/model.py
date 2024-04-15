from flask import Flask, request, render_template
import kepco_model.keras as model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    days = int(request.form['days'])
    prediction = model.predict(days)
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)