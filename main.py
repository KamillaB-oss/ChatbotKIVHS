from training import get_response

from flask import Flask, render_template, request, jsonify

from main import get_response

#app wird initialisert
app = Flask(__name__)
# hier wird die erstellte HTML Datei dem localhost zugewiesen
@app.route("/")
def home():
    return render_template('chat.html')
# bei der Eingabe wird hier die programmierte Funktion aus training.py aufgerufen, um die Antwort für die Frage zu erhalten
@app.route("/chatbot", methods = ["GET", "POST"])
def chatbot():
    user_input = request.form["message"]
    return str(get_response(str(user_input)))

# führt die Webanwendung aus
if __name__ == '__main__':
    app.run()
