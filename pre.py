import numpy as np
import pandas as pd
from flask import Flask,request,render_template

#import pickle
import joblib

app = Flask(__name__,template_folder='template')
model=joblib.load("student_mark_predictor.pkl")

@app.route('/')
def home():
    return render_template('index.html')

df=pd.DataFrame()
@app.route('/predict',methods=['POST'])
def predict():
    if (request.method) =='POST':
        global df
        data1 = []
        data = request.form.get('mark')
        print(f"Data = {data}")
        # data = int(data)

        if int(data) >= 1 and int(data) <=24:

            data1.append(int(data))

            features = np.array(data1)

            output = model.predict( [features] )[0][0].round(2)

            if output <= 100:

                df = pd.concat( [df , pd.DataFrame( {'Study Hours' : data , 'Predicted Marks ' : [output]} )] ,ignore_index=True )
                print(df)
                df.to_csv('Predicted_data.csv')

                return render_template('index.html',predict = f'If You study {data} hours then You can get {output}% Marks')
            else:
                return render_template('index.html',predict = f'If You study {data} hours then You can get 100% Marks')
        else:
            return render_template('index.html',predict = 'Please Enter hours between 1-24')



if __name__ == '__main__':
    app.debug=True
    app.run()
