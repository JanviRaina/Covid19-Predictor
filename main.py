from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()

@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method=='POST': 
        myDict=request.form
        has_fever=int(myDict['fever'])
        person_age=int(myDict['age'])
        body_pain=int(myDict['pain'])
        runny_nose=int(myDict['runnyNose'])
        difficult_Breathe=int(myDict['diffbreath'])
        travelhistory=int(myDict['travelHistory'])
        inputFeatures=[has_fever,body_pain,person_age,runny_nose,difficult_Breathe]
        infProb=clf.predict_proba([inputFeatures])[0][1]
        print(infProb)

        return render_template('show.html',inf=round(infProb*100))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
