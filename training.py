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
    for intent in data ["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #print(words)

    #die Methode kann keine konkreten Wörter identifizieren, deshalb wird eine Liste zurückgegeben, wie oft das ein Wort in den Trainingsdaten vorkommt

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("Data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# ''try:
#     model.load("model.tflearn")
# except:''
    #n_epoch bedeutet wie oft das Modell die Daten sehen wird
model.fit(training, output, n_epoch = 1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
    # Damit das Modell nicht jedes mal abgespielt werden muss, kann die gespeicherte Version verwendet werden

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def get_response(msg):
    results = model.predict([bag_of_words(msg, words)])[0]
# results gibt für jeden Neuronen die Wahrscheinlichkeit ein, die für die Eingabe des Nutzers zu trifft und mit argmax soll den Tag mit der größten Wahrscheinlichkeit ausgewählt werden
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.8:
    # nun wird der konkrete Tag ausgegeben
            #print(tag)
    # damit eine konkrete Antwort ausgewählt werden kann, wird die Json Datei geöffnet und ein random response für den ausgewählten Tag als Antwort zurückgegeben
        for tg in data["intents"]:
            if tg['tag'] == tag:
                response = tg['responses']
            #print(random.choice(responses))
        response_uni = [ftfy.fix_text(satz) for satz in response]
        return str(random.choice(response_uni))
            #resp = random.choice(response_uni)
            #return str(resp)
    else:
        return "I did't unterstand the Question. Could you ask anathor one?"
    
# print("Let's chat! (type 'quit' to exit)")
# #print(data["intents"])
# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

#     resp = get_response(sentence)
#     print(resp)