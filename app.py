import os
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model_path = 'lstm_word_completion.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

tokenizer_path = 'tokenizer.pickle'
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    raise FileNotFoundError(f"Tokenizer file '{tokenizer_path}' not found.")

index_word = {index: word for word, index in tokenizer.word_index.items()}

vocab_size = len(tokenizer.word_index) + 1
max_len = 20

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/complete', methods=['POST'])
def complete_sentence():
    data = request.json
    seed = data.get('seed', '')
    
    try:
        num_words = int(data.get('num_words', 1))
    except ValueError:
        return jsonify({"error": "num_words must be an integer."}), 400

    print(f"Received seed: {seed}, num_words: {num_words}")

    if not seed or num_words <= 0:
        return jsonify({"error": "Invalid input."}), 400

    seed_words = seed.split()
    if len(seed_words) > max_len - 1:
        seed = ' '.join(seed_words[-(max_len - 1):])

    try:
        generated = generate_text(seed, model, num_words)
        return jsonify({"generated": generated})
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        return jsonify({"error": "Error generating text."}), 500

def generate_text(seed, model, num_words):
    """
    Generates a sequence of words based on a seed text using the trained model.
    
    Parameters:
    - seed: The initial text input to begin generation.
    - model: The LSTM model used for generating text.
    - num_words: The number of words to generate.

    Returns:
    - A string containing the seed text followed by the generated words.
    """
    for _ in range(num_words):
        tokens = tokenizer.texts_to_sequences([seed])[0]
        tokens = pad_sequences([tokens], maxlen=max_len-1, padding='pre')
        preds = model.predict(tokens, verbose=0)
        next_idx = np.argmax(preds)
        next_word = next_word_from_index(next_idx)
        if next_word:
            seed += ' ' + next_word
    return seed.strip()

def next_word_from_index(index):
    """
    Retrieves the next word based on its index from the mapping.

    Parameters:
    - index: The index of the word to retrieve.

    Returns:
    - The corresponding word or "unknown" if the index is not found.
    """
    return index_word.get(index + 1, "unknown")

if __name__ == '__main__':
    app.run(debug=True)
