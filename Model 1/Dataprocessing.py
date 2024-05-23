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
count = 1
print("count: "+count)
count+=1

# Construct the absolute path to the CSV file
file_path = os.path.join(base_dir, 'diets/All_Diets.csv')
print("count: "+count)
count+=1

# Load the dataset
df = pd.read_csv(file_path)
print("count: "+count)
count+=1

# Inspect the dataset
print(df.head())
print(df.columns)
print(df['Diet_type'].unique())  # Check the unique diet labels
print("count: "+count)
count+=1

# Example column names in the dataset
recipes = df['Recipe_name']  # assuming the dataset has a column 'Recipe_name'
diet_labels = df['Diet_type']  # assuming the dataset has a column 'Diet_type'
print("count: "+count)
count+=1

# Tokenizing the text data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(recipes)
word_index = tokenizer.word_index
print("count: "+count)
count+=1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(recipes)
padded_sequences = pad_sequences(sequences, padding='post')
print("count: "+count)
count+=1

# Convert diet labels to categorical data
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(diet_labels)
label_sequences = label_tokenizer.texts_to_sequences(diet_labels)
label_sequences = np.array(label_sequences).flatten()
print("count: "+count)
count+=1

# Get the number of unique labels
num_classes = len(label_tokenizer.word_index) + 1
print("count: "+count)
count+=1

# Build the RNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=padded_sequences.shape[1]),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
print("count: "+count)
count+=1

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

print("count: "+count)
count+=1

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, label_sequences, test_size=0.2, random_state=42)
print("count: "+count)
count+=1
# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
print("count: "+count)
count+=1
# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')
print("count: "+count)
count+=1
# Example input for prediction
new_recipes = ["list of food items here"]
new_sequences = tokenizer.texts_to_sequences(new_recipes)
new_padded_sequences = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1], padding='post')

predictions = model.predict(new_padded_sequences)
predicted_labels = np.argmax(predictions, axis=1)
print("count: "+count)
count+=1
# Convert numeric labels back to diet types
predicted_diets = label_tokenizer.sequences_to_texts(predicted_labels.reshape(-1, 1))
print(predicted_diets)
print("count: "+count)
count+=1
# Save the model
model.save('diet_recipe_rnn_model.h5')
