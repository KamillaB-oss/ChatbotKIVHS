import nltk
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle
import ftfy

with open("intents.json") as file:
    data = json.load(file)


try:
#rb = read Bytes
# falls etwas bei der intents.json Datei geändert worden ist, dann einfach in den try ein x schreiben, damit das nicht ausgeführt wird oder man einfach den pickle File löschen
    with open("Data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

#Intents ist die komplette Dateistruktur der JSON Datei
#patterns sind die möglichen Fragestellungen
#tag beschreibt eine Gruppe in ähnliche Fragen gestellt werden
# Es durch die JSON Datei im Array Intents durchiteriert
    for intent in data ["intents"]:
        for pattern in intent["patterns"]:
            # die vordefinierten Fragen werden in einzelne Wörter aufgesplittet
            wrds = nltk.word_tokenize(pattern)
            # die einzelnen Wörter werden in die Liste Word hinzugefügt, damit festgehalten werden kann, welche Wörter der Bot schon kennt
            words.extend(wrds)
            # die Liste erhält alle Eingabes#tze
            docs_x.append(wrds)
            # zu den jewieligen Eingabesätzen, wird in dieser Liste hinzugefügt, zu welcher Gruppe von Fragen das zugehörig ist
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            # hier werden alle sich in der JSON Datei befindeten Tags gespeichert
            labels.append(intent["tag"])
    # die sich in der Liste befindeten Wörter werden klein geschrieben und zu ihrem Wortbaustamm zurückgeführt, außerdem sollen ? ignoriert werden
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # Duplikate werden entfert und die Liste wird sortiert
    words = sorted(list(set(words)))
    # Liste wird sortiert
    labels = sorted(labels)

    #print(words)

    #die Methode kann keine konkreten Wörter identifizieren, deshalb wird eine Liste zurückgegeben, wie oft das ein Wort in den Trainingsdaten vorkommt

    training = []
    output = []
    #Liste soll mit 0 gefüllt und so lange wie die Liste labels sein
    out_empty = [0 for _ in range(len(labels))]
    # x repräsentiert den Index des Mustersatzes in doca_x und doc repräsentiert den Mustersatz
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]
        # für jeden Wort, welches in words vorkommt wird die Liste Bag um 1 erweitert und kommt ein Wort nicht vor, so wird die Liste um 0 erweitert
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        # die Liste training enthält die Werte der Liste ba
        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    # die Werte und Listen werden gespeichert, damit der ganze obere Code nicht bei jeden ausführen des Programms durchlaufen muss (bei Änderung des Datensatzes kann die pickle Datei gelöscht werden, dann wird automatisch alles nichmal ausgeführt)
    with open("Data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
# Model wird ausgesucht und trainiert
tf.compat.v1.reset_default_graph()
# es wird das neuronale Netz ausgewählt, um den Chatbot zu trainieren. Es wird ein Input Layer, drei Hidden Layer und ein output Layer implementiert
net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)
# Model wird angewandt
model = tflearn.DNN(net)

# ''try:
#     model.load("model.tflearn")
# except:''
    #n_epoch bedeutet wie oft das Modell die Daten sehen wird
model.fit(training, output, n_epoch = 1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
    # Damit das Modell nicht jedes mal abgespielt werden muss, kann die gespeicherte Version verwendet werden
# macht dasselbe wie mit den vorhandenen Datensatz, er nimmt die Eingabe und untersucht welche Wörter wie oft vorkommen
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)
#hier wird die Ausgabe des Chatbots erzeigt
def get_response(msg):
    results = model.predict([bag_of_words(msg, words)])[0]
# results gibt für jeden Neuronen die Wahrscheinlichkeit ein, die für die Eingabe des Nutzers zu trifft und mit argmax soll den Tag mit der größten Wahrscheinlichkeit ausgewählt werden
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.8:
    # nun wird der konkrete tag ausgegeben
    # damit eine konkrete Antwort ausgewählt werden kann, wird die Json Datei geöffnet und ein random response für den ausgewählten Tag als Antwort zurückgegeben
        for tg in data["intents"]:
            if tg['tag'] == tag:
                response = tg['responses']
        # es wurde ftfy verwendet, damit ä ü ö ß ohne Probleme ausgegeben werden können
        # es wird eine Antwort zurückgegeben
        response_uni = [ftfy.fix_text(satz) for satz in response]
        return str(random.choice(response_uni))
    else:
        # falls die Frage nicht einem tag zugewiesen werden könnte oder die Zutrefferwahrscheinlichkeit zu gering ist, wird dieser Satz zurückgegeben
        return "Ich habe diese Frage leider nicht verstanden. Könnten Sie mir bitte eine neue Frage stellen?"
