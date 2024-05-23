import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
from sklearn.model_selection import train_test_split
import os

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# Construct the absolute path to the CSV file
file_path = os.path.join(base_dir, 'TasteTrios - Sheet1.csv')

# Load the dataset
df = pd.read_csv(file_path)

# Inspect the dataset
print(df.head())
print(df.columns)
print(df.info())

# Example column names in the dataset
# Adjust based on actual column names
food_items = df['Food_Item']  # Assuming the dataset has a column 'Food_Item'
taste_trios = df['Taste_Trio']  # Assuming the dataset has a column 'Taste_Trio'

# Tokenizing the text data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(food_items)
word_index = tokenizer.word_index

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(food_items)
padded_sequences = pad_sequences(sequences, padding='post')

# Convert taste trios labels to categorical data
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(taste_trios)
label_sequences = label_tokenizer.texts_to_sequences(taste_trios)
label_sequences = np.array(label_sequences).flatten()

# Get the number of unique labels
num_classes = len(label_tokenizer.word_index) + 1

# Build the RNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=padded_sequences.shape[1]),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, label_sequences, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Example input for prediction
new_food_items = ["list of food items here"]
new_sequences = tokenizer.texts_to_sequences(new_food_items)
new_padded_sequences = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1], padding='post')

predictions = model.predict(new_padded_sequences)
predicted_labels = np.argmax(predictions, axis=1)

# Convert numeric labels back to taste trios
predicted_taste_trios = label_tokenizer.sequences_to_texts(predicted_labels.reshape(-1, 1))
print(predicted_taste_trios)

# Save the model
model.save('taste_trio_rnn_model.h5')
