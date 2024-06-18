import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import tkinter as tk
from tkinter import *

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def send_message():
    msg = entry_field.get()
    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n', 'user')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(tk.END, "Alita: " + res + '\n\n', 'bot')

        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)
        entry_field.delete(0, tk.END)

def on_enter_key(event):
    send_message()

base = tk.Tk()
base.title("Alita")
base.geometry("400x500")
base.resizable(width=True, height=True)

# Create Chat window
ChatLog = tk.Text(base, bd=0, bg="white", height="8", width="30", font="Arial")
ChatLog.config(state=tk.DISABLED)

# Create tags for text alignment
ChatLog.tag_configure('user', justify='right')
ChatLog.tag_configure('bot', justify='left')

# Bind scrollbar to Chat window
scrollbar = tk.Scrollbar(base, command=ChatLog.yview, cursor="Arrow")
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
ChatLog['yscrollcommand'] = scrollbar.set

# Create Entry field for user input
entry_field = tk.Entry(base, font=("Verdana", 12), width="29")
entry_field.bind("<Return>", on_enter_key)  # Bind Enter key to send message

# Create Button to send message
SendButton = tk.Button(base, font=("Verdana", 12, 'bold'), text="Send", width="10", height=2,
                        bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                        command=send_message)

# Layout widgets
ChatLog.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
entry_field.pack(side=tk.LEFT, pady=10, padx=10, fill=tk.X, expand=True)
SendButton.pack(side=tk.RIGHT, pady=10, padx=10)

base.mainloop()
