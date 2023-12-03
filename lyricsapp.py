import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import json

# Load the model
model = load_model('song_lyrics_generator.h5')

# Load the tokenizer
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Ensure max_sequence_len matches the value used during training
max_sequence_len = 1139  # Replace this with the correct value used during training

# Function to generate lyrics
def generate_lyrics_to_continue(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')  # Use max_sequence_len here
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Streamlit UI
st.title('Song Lyrics Generator')

# Input seed text
seed_text = st.text_input('Enter Seed Text:', 'You are my King')

# Input number of words to generate
next_words = st.slider('Number of Words to Generate:', min_value=5, max_value=100, value=20)

# Generate lyrics button
if st.button('Generate Lyrics'):
    generated_lyrics = generate_lyrics_to_continue(seed_text, next_words)
    st.text('Generated Lyrics:')
    st.write(generated_lyrics)
