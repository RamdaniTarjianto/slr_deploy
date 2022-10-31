import requests
import tensorflow
import urllib.request
from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras_preprocessing.sequence import pad_sequences
import pickle

# import logging
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

np.random.seed(1337)

# graph = tf.get_default_graph()

model = tensorflow.keras.models.load_model("model_fasttext_lstm.h5")
# label = ["Hourse", "Human"]

#star Flask application
app = Flask(__name__)

#load tokenizer pickle file
with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

def preprocess_text(texts, max_review_length = 120):
    #tok = Tokenizer(num_words=num_max)
    #tok.fit_on_texts(texts)
    lstm_texts_seq = tok.texts_to_sequences(texts)
    lstm_texts_mat = pad_sequences(lstm_texts_seq, maxlen=max_review_length)
    return lstm_texts_mat

@app.route('/predict',methods=['POST'])

def predict():
    text = request.args.get('text')
    x = preprocess_text([text])
    # with graph.as_default():
    #     y = int(np.round(model.predict(x)))
    #     if y == 1:
    #         return jsonify({'prediction': "Include"})
    #     else:
    #         return jsonify({'prediction': "Exclude"})
    predictions = model.predict(x)
    print(predictions)
    # predicted_class_indices = np.where(predictions < 0.5, 0, 1)
    if predictions > 0.5:
        value = "include"
    else:
        value = "exclude"
    return value

if __name__ == "__main__":
    # Run locally
    app.run(debug=False)
