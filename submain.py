import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random


conversations = [
    ("Hello", "Hi there!"),
    ("How are you?", "I'm good. How about you?"),
    ("What's your name?", "I'm a chatbot."),
    ("Goodbye", "Goodbye!"),
]

# Separate the input and target responses
input_text = [x[0] for x in conversations]
target_text = [x[1] for x in conversations]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_text + target_text)

input_sequences = tokenizer.texts_to_sequences(input_text)
target_sequences = tokenizer.texts_to_sequences(target_text)

# Pad sequences to the same length
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')


vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
    LSTM(256, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')
    
    predicted_sequence = []
    
    for _ in range(max_sequence_length):
        predicted_probabilities = model.predict(input_seq)[0]
        predicted_id = np.argmax(predicted_probabilities)
        
        if predicted_id == 0:
            break
        
        predicted_word = tokenizer.index_word.get(predicted_id, '')
        predicted_sequence.append(predicted_word)
        
        input_seq[0, -1] = predicted_id
    
    return ' '.join(predicted_sequence)

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = generate_response(user_input)
    print("Chatbot:", response)
