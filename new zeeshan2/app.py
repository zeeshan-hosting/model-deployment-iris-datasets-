from flask import Flask,render_template,request
import pickle
import numpy as np
__model=None


app=Flask(__name__)


@app.route('/')
def man():
    global _model
    if __model is None:
        with open('model.pickle', 'rb') as f:
            _model = pickle.load(f)
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
   
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred =_model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
