from flask import Flask, render_template, request
import pickle

app= Flask(__name__)

model=pickle.load(open('saved_model.sav','rb'))

@app.route('/')
def home():
    result=''
    return render_template('index.html',**locals())

@app.route('/predict',methods=['Post','GET'])
def predict():
    sepal_length=float(request.form['Sepal_Length'])
    sepal_width=float(request.form['Sepal_width'])
    petal_length=float(request.form['Petal_Length'])
    petal_width=float(request.form['Petal_width'])
    result=model.predict([[sepal_length,sepal_width,petal_length,petal_width ]])[0]
    return render_template('index.html',**locals())

if __name__ == '__main__':
    app.run(debug=True)




