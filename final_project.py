# -*- coding: utf-8 -*-
"""Final_Project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vaVvMrhz2yMm4EgeJYGrC_scfKl4YWcr
"""

pip install langdetect

"""# Import necessary libraries"""

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import json
import tensorflow as tf
import tensorflow.keras as keras
!pip install scikeras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Embedding, Dense, Dropout,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import load_model
from langdetect import detect
from google.colab import drive
drive.mount('/content/drive')

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/AI/lyrics-data.csv')

data.head()

#Printing the size of dataset
print("Size of Dataset:",data.shape)

"""# Filter only English lyrics"""

# Filter only English lyrics
data = data[data['language'] == 'en']
data

"""# Dropping all columns except the lyrics colunm"""

drop_features = ['ALink', 'SName', 'SLink', 'language']
data.drop(drop_features, axis = 1, inplace = True)
data

data.head()

# Limiting the 'data' variable to the first 200 elements
data = data[:200]

# shape
data.shape

"""# Tokenization"""

# Create a Tokenizer object
tokenizer = Tokenizer()

# Fit the tokenizer on the text data in the 'Lyric' column, converting to lowercase
tokenizer.fit_on_texts(data['Lyric'].astype(str).str.lower())

# Get the total number of unique words in the dataset + 1 (to account for padding)
total_words = len(tokenizer.word_index) + 1

# Convert the text in the 'Lyric' column to sequences of corresponding tokens
tokenized_sentences = tokenizer.texts_to_sequences(data['Lyric'].astype(str))

# Display the tokenized representation of the first sentence in the dataset
tokenized_sentences[0]

# Retrieve the word index mapping from the tokenizer
word_index = tokenizer.word_index

# Display or further use the obtained word index
word_index

"""#Data Preprocessing"""

input_sequences = list()
for i in tokenized_sentences:
    for t in range(1, len(i)):
        n_gram_sequence = i[:t+1]
        input_sequences.append(n_gram_sequence)
# Pre padding
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

"""# Feature Engineering"""

# Feature Engineering: One-hot encoding for categorical crossentropy loss
X, labels = input_sequences[:,:-1],input_sequences[:,-1]
y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

"""# Training the Model"""

# Create the model
model = Sequential()

# Embedding layer to represent words as vectors
model.add(Embedding(input_dim=total_words, output_dim= 50, input_length=max_sequence_len-1))

# Bidirectional LSTM layer for capturing context from both directions
model.add(Bidirectional(LSTM(100)))

# Dropout layer to prevent overfitting.
model.add(Dropout(0.1))

# Dense layer with softmax activation for output probabilities
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping to prevent overfitting.
earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1, callbacks=[earlystop], validation_data=(X_test, y_test))

"""# Model Evaluation"""

# Evaluate the model on the test set
evaluation = model.evaluate(X_test, y_test)

# Extract relevant metrics
loss, accuracy = evaluation[0], evaluation[1]

# Print the evaluation results
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""# Training the Model using GridSearchCV"""

# Function to create the model
def create_model(embedding_dim=50, lstm_units=50, dropout_rate=0.1, dense_units1=256, dense_units2=128):
    model = Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create a KerasClassifier with the create_model function
keras_classifier = KerasClassifier(build_fn=create_model, epochs=10, batch_size=128, verbose=1)

# Define the hyperparameter grid
param_grid = {
    'model__embedding_dim': [50],
    'model__lstm_units': [50],
    'model__dropout_rate': [0.1],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, cv=3)

# Fit the model to the data
grid_search_result = grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search_result.best_params_)

# Get the best model
best_model = grid_search_result.best_estimator_.model

# Evaluate the best model on the test set
test_evaluation = best_model.evaluate(X_test, y_test)

# Extract relevant metrics
test_loss, test_accuracy = test_evaluation[0], test_evaluation[1]

# Print the test set performance
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')



"""# Function to generate lyrics"""

# Function to generate lyrics
def generate_lyrics_to_continue(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example usage of the generate_lyrics function
generated_lyrics = generate_lyrics_to_continue("Sing to me", 20)
generated_lyrics

"""#Saving the Model and the Tokenizer"""

# Save model
model.save('song_lyrics_generator.h5')

# Save tokenizer to a file
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))