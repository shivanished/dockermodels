import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json  # Import json to save tokenizer indices and label encodings

# Function to save tokenizer index and label encoder classes
def save_json(filepath, obj):
    with open(filepath, 'w') as file:
        json.dump(obj, file)

# Get the directory of the current script
base_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the CSV file
file_path = os.path.join(base_dir, 'TasteTrios - Sheet1.csv')
df = pd.read_csv(file_path)

# Combine the ingredient columns into a single column
df['Combined_Ingredients'] = df['Ingredient 1'].astype(str) + ' ' + df['Ingredient 2'].astype(str) + ' ' + df['Ingredient 3'].astype(str)
inputs = df['Combined_Ingredients']
labels = df['Classification Output']

# Tokenizing the text data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(inputs)
word_index = tokenizer.word_index

# Save the tokenizer index
save_json('taste_trio_tokenizer_word_index.json', tokenizer.word_index)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(inputs)
padded_sequences = pad_sequences(sequences, padding='post')

# Convert labels to integers using LabelEncoder
label_encoder = LabelEncoder()
label_sequences = label_encoder.fit_transform(labels)

# Save the label encoder classes
save_json('taste_trio_label_encoder_classes.json', list(label_encoder.classes_))

# Get the number of unique labels
num_classes = len(label_encoder.classes_)

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
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Example input for prediction
new_ingredients = ["Pumpkin Cinnamon Ginger"]
new_sequences = tokenizer.texts_to_sequences(new_ingredients)
new_padded_sequences = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1], padding='post')
predictions = model.predict(new_padded_sequences)
predicted_labels = np.argmax(predictions, axis=1)
predicted_diets = label_encoder.inverse_transform(predicted_labels)

print(f'Predicted Classification Output for "{new_ingredients[0]}": {predicted_diets[0]}')

# Save the model
model.save('taste_trio_rnn_model.h5')
