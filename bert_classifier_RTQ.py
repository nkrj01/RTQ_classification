import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import tensorflow_hub as hub
import tensorflow as tf
from sklearn import preprocessing
import tensorflow_text as txt
from sklearn import tree
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential, regularizers
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

# import the text data
df = pd.read_excel("RTQ_others.xlsx")
# select relevant columns
df2 = df[['Text', 'PDE']]

# select all the rows using the relevant labels
all_labels = pd.read_excel("lable_list").to_list()
include_labels = all_labels
drop_labels = [lable for lable in all_labels if lable not in include_labels]
df2 = df2[df2["PDE"].isin(drop_labels) == False]

# define preprocessor and encoder
bert_preprocessor = hub.KerasLayer(preprocess_url)
bert_encoder = hub.KerasLayer(encoder_url)

# get X and y for test and cross validation
X = df2["Text"]
y_temp = df2['PDE']
X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2, stratify=y_temp)

# get embeddings
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocessor(text_input)
outputs = bert_encoder(preprocessed_text)

# define NN model structure 
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs["pooled_output"])
l = tf.keras.layers.Dense(80, activation='relu', name="hidden1")(l)
l = tf.keras.layers.Dense(40, activation='relu', name="hidden2")(l)
l = tf.keras.layers.Dense(30, activation='relu', name="hidden3")(l)
l = tf.keras.layers.Dense(10, activation='relu', name="hidden4")(l)
l = tf.keras.layers.Dense(8, activation='softmax', name="output")(l)

model = tf.keras.Model(inputs=[text_input], outputs=[l])

METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
]

model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=METRICS
)

# fit and evaluate
model.fit(X_train, y_train, epochs=100, batch_size=32)
model.evaluate(X_test, y_test)
