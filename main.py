from training import get_response
import nltk
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from flask import Flask, render_template, request, jsonify

from main import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('chat.html')

@app.route("/chatbot", methods = ["GET", "POST"])
def chatbot():
    user_input = request.form["message"]
    return str(get_response(str(user_input)))


if __name__ == '__main__':
    app.run()
