import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from getmail import send_mail
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from nltk. tokenize import word_tokenize, sent_tokenize
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"
app.config["DEBUG"] = True

class_names = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
model = tf.keras.models.load_model("abstract_simplifier", custom_objects = {"TextVectorization": TextVectorization})

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/simplifier')
def simplifier():
  return render_template('simplifier.html') 

@app.route('/simplify', methods = ["GET", "POST"])
def simplify():
      
  if request.method == 'POST':
    abstract = request.form["abstract"]

  ret = """<!DOCTYPE html>
<head>
   <title>Context Classification</title>
</head>
<body>"""

  Background, Objective, Methods, Results, Conclusions = [], [], [], [], []

  sentences = abstract.strip('.').split('.')

  lines = [[i, len(sentences)] for i in range(len(sentences))]

  preds = model.predict(x = (tf.constant(lines), tf.constant(sentences)))

  final_preds = tf.argmax(preds, axis = 1)

  for ind, pred in enumerate(final_preds.numpy()):
    if pred == 0:
      Background.append(sentences[ind])
    elif pred == 1:
      Conclusions.append(sentences[ind])
    elif pred == 2:
      Methods.append(sentences[ind])
    elif pred == 3:
      Objective.append(sentences[ind])
    elif pred == 4:
      Results.append(sentences[ind])

  if Background:
    ret += f"\n<h1>Background</h1>\n<p>{'.'.join(Background)}</p>"

  if Objective:
    ret += f"\n<h1>Objective</h1>\n<p>{'.'.join(Objective)}</p>"

  if Methods:
    ret += f"\n<h1>Methods</h1>\n<p>{'.'.join(Methods)}</p>"
  
  if Results:
    ret += f"\n<h1>Results</h1>\n<p>{'.'.join(Results)}</p>"
  
  if Conclusions:
    ret += f"\n<h1>Conclusions</h1>\n<p>{'.'.join(Conclusions)}</p>"

  ret += "\n</body>"

  return ret

@app.route('/summarizer')
def summarizer():
  return render_template('summarizer.html') 

@app.route('/summarize', methods = ["GET", "POST"])
def summarize():
      
  if request.method == 'POST':
    abstract = request.form["abstract"]

  ret = """<!DOCTYPE html>
<head>
   <title>Summarization</title>
</head>
<body>"""

  summary = sumar(abstract)
  ret += f"\n<h1>Summary</h1>\n<p>{summary}</p>"
  ret += "\n</body>"

  return ret

@app.route('/simsumer')
def simsumer():
  return render_template('simsumer.html')

@app.route('/simsum', methods = ["GET", "POST"])
def simsum():
      
  if request.method == 'POST':
    abstract = request.form["abstract"]

  ret = """<!DOCTYPE html>
<head>
   <title>Classification and summary</title>
</head>
<body>"""

  Background, Objective, Methods, Results, Conclusions = [], [], [], [], []

  sentences = abstract.strip('.').split('.')

  lines = [[i, len(sentences)] for i in range(len(sentences))]

  preds = model.predict(x = (tf.constant(lines), tf.constant(sentences)))

  final_preds = tf.argmax(preds, axis = 1)

  for ind, pred in enumerate(final_preds.numpy()):
    if pred == 0:
      Background.append(sentences[ind])
    elif pred == 1:
      Conclusions.append(sentences[ind])
    elif pred == 2:
      Methods.append(sentences[ind])
    elif pred == 3:
      Objective.append(sentences[ind])
    elif pred == 4:
      Results.append(sentences[ind])

  if Background:
    background = sumar('.'.join(Background))
    
    if background != '': 
      ret += f"\n<h1>Background</h1>\n<p>{background}</p>"

  if Objective:
    objective = sumar('.'.join(Objective))
    
    if objective != '': 
      ret += f"\n<h1>Objective</h1>\n<p>{objective}</p>"

  if Methods:
    methods = sumar('.'.join(Methods))
    
    if methods != '': 
      ret += f"\n<h1>Methods</h1>\n<p>{methods}</p>"
  
  if Results:
    results = sumar('.'.join(Results))
    
    if results != '': 
      ret += f"\n<h1>Results</h1>\n<p>{results}</p>"
  
  if Conclusions:
    conclusions = sumar('.'.join(Conclusions))
    
    if conclusions != '': 
      ret += f"\n<h1>Conclusions</h1>\n<p>{conclusions}</p>"

  ret += "\n</body>"

  return ret

def sumar(text):
  stopWords = set (stopwords.words ("english"))
  words = word_tokenize(text)

  freqTable = dict() 
  for word in words:
    word = word.lower()
    if word in stopWords:
      continue 
    if word in freqTable:
      freqTable[word] += 1
    else:
      freqTable[word] = 1
      
  sentences = sent_tokenize(text)
  sentenceValue = dict()

  for sentence in sentences: 
    for word, freq in freqTable.items(): 
      if word in sentence.lower(): 
        if sentence in sentenceValue:
          sentenceValue[sentence] += freq
        else:
          sentenceValue[sentence] = freq

  sumValues = 0
  for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]

  average = int(sumValues / len( sentenceValue))

  summary = ''
  for sentence in sentences: 
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (0.5 * average)):
      summary += " " + sentence

  return summary

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/sentsafe',methods=['GET', 'POST'])
def send_sentsafe():
    if request.method == 'POST':
        email = request.form['email']
        comments = request.form['comments']
        name=request.form['name']
        comments=email+"  \n "+name+"  \n "+comments
        send_mail(email,comments)
    return render_template('sentfeed.html')

if __name__=="__main__":
    app.run(debug=True)