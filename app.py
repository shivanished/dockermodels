from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

app = Flask(__name__)

# Load models (adjust paths as necessary)
model1 = tf.keras.models.load_model('/app/Model1/diet_recipe_rnn_model.h5')
model2 = tf.keras.models.load_model('/app/Model2/taste_trio_rnn_model.h5')

# Initialize tokenizers
# These should be the same tokenizers used during training of the models
diet_tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
taste_trio_tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

# Load the tokenizer word indices (adjust paths as necessary)
diet_tokenizer.word_index = load_word_index('/app/Model1/diet_tokenizer_word_index.json')
taste_trio_tokenizer.word_index = load_word_index('/app/Model2/taste_trio_tokenizer_word_index.json')

def load_word_index(filepath):
    import json
    with open(filepath, 'r') as f:
        word_index = json.load(f)
    return word_index

@app.route('/generate_meal_plan', methods=['POST'])
def generate_meal_plan():
    data = request.json
    ingredients = data['ingredients']
    
    # Preprocess ingredients for Model 1
    diet_sequences = diet_tokenizer.texts_to_sequences(ingredients)
    diet_padded_sequences = pad_sequences(diet_sequences, padding='post')
    
    # Predict diet categories using Model 1
    diet_predictions = model1.predict(diet_padded_sequences)
    diet_labels = np.argmax(diet_predictions, axis=1)
    
    # Format the output for Model 2
    ingredient_combos = format_for_taste_trio(ingredients, diet_labels)
    
    # Preprocess data for Model 2
    taste_trio_sequences = taste_trio_tokenizer.texts_to_sequences(ingredient_combos)
    taste_trio_padded_sequences = pad_sequences(taste_trio_sequences, padding='post')
    
    # Predict taste trios using Model 2
    taste_trio_predictions = model2.predict(taste_trio_padded_sequences)
    taste_trio_labels = np.argmax(taste_trio_predictions, axis=1)
    
    # Convert predictions to readable format
    final_meal_plan = taste_trio_tokenizer.sequences_to_texts(taste_trio_labels.reshape(-1, 1))
    
    return jsonify({'meal_plan': final_meal_plan})

def format_for_taste_trio(ingredients, diet_labels):
    # This function formats the output of Model 1 to be used as input for Model 2
    # Adjust according to your needs
    formatted_output = []
    for ingredient, label in zip(ingredients, diet_labels):
        formatted_output.append(f"{ingredient} (diet label: {label})")
    return formatted_output

#@app.route('/predict_diet', methods=['POST'])
def predict_diet():
    data = request.json
    ingredients = data['ingredients']
    
    # Preprocess data for Model 1
    diet_sequences = diet_tokenizer.texts_to_sequences(ingredients)
    diet_padded_sequences = pad_sequences(diet_sequences, padding='post')
    
    # Predict diet categories using Model 1
    prediction = model1.predict(diet_padded_sequences)
    diet_labels = np.argmax(prediction, axis=1)
    
    return jsonify({'prediction': diet_labels.tolist()})

#@app.route('/predict_taste_trio', methods=['POST'])
def predict_taste_trio():
    data = request.json
    ingredient_combo = data['ingredient_combo']
    
    # Preprocess data for Model 2
    taste_trio_sequences = taste_trio_tokenizer.texts_to_sequences(ingredient_combo)
    taste_trio_padded_sequences = pad_sequences(taste_trio_sequences, padding='post')
    
    # Predict taste trios using Model 2
    prediction = model2.predict(taste_trio_padded_sequences)
    taste_trio_labels = np.argmax(prediction, axis=1)
    
    return jsonify({'prediction': taste_trio_labels.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)

