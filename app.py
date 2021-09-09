from flask import Flask, app, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    model = joblib.load('my_model.pkl')

    if request.method == 'POST':
        my_features = [x for x in request.form.values()]

    
    final_features = np.asarray(my_features).reshape(1,-1)
    preds = model.predict(final_features)
    

    return render_template('display.html',preds = preds)


if __name__ == '__main__':
    app.run(debug=True)