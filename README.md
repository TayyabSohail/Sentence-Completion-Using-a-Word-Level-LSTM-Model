# Sentence-Completion-Using-a-Word-Level-LSTM-Model

## RESEARCH PAPER
[Report_Word_Completion_using_LSTM.pdf](https://github.com/user-attachments/files/17525796/Report_Word_Completion_using_LSTM.pdf)


# Project Overview
This project involves building a word-level LSTM (Long Short-Term Memory) model for sentence completion, trained on Shakespeare's plays. The goal is to predict the next word in a sequence, providing real-time word suggestions as users type. Users can input partial sentences, and the trained model will suggest the next words dynamically. The interface provides an interactive experience for exploring how well the model can generate text in the style of Shakespeare.

Additionally, various hyperparameter settings (like learning rate, batch size, and model depth) were tested to assess their impact on the quality, coherence, and fluency of the generated sentences.

# Objectives

Build a word-level LSTM model capable of predicting the next word in a sequence.

Train the model on Shakespeare's plays to mimic the literary style.

Create a user-friendly interface where users can input partial sentences and receive real-time word suggestions.

Experiment with different hyperparameters to improve model performance.

Evaluate the coherence and fluency of the generated sentences.

# Tools and Technologies Used

Programming Language: Python

# Libraries:

TensorFlow/Keras: For building and training the LSTM model.

NumPy: For numerical operations and data manipulation.

pandas: For loading and handling the Shakespeare dataset.

Flask/Django: (Optional) For developing a web-based user interface.

Pickle: For saving and loading the trained tokenizer.

matplotlib/Seaborn: For plotting training progress (optional).

# Dataset
The dataset used is a collection of Shakespeare's plays, which includes texts such as Hamlet, Macbeth, and Romeo and Juliet. You can download it from this Kaggle link.

# Model Architecture

Embedding Layer: To represent words in a dense vector space.

Bidirectional LSTM Layer: For capturing both past and future dependencies in the text.

Dropout Layer: To prevent overfitting.

Dense Layer: To predict the next word by outputting a probability distribution over the vocabulary.

# Training Details

Input: Tokenized sequences from Shakespeare’s plays.

Output: Predicted word for a given input sequence.

Loss Function: Categorical cross-entropy (for multi-class classification).

Optimizer: Adam (adaptive learning rate).

Batch Size: 1024 (can be adjusted based on system capacity).

Epochs: 20 (with early stopping based on loss).

# How to Run the Project

## 1. Prerequisites

Python 3.x

Install required packages:
pip install tensorflow pandas numpy

## 2. Dataset Preparation

Download the Shakespeare dataset from Kaggle and save it as a CSV file (Shakespeare_data.csv).

Ensure the dataset has a column called PlayerLine which contains the lines spoken by characters in Shakespeare's plays.

## 3. Running the Code

Load and Preprocess Data: The text is cleaned by removing punctuation and converting it to lowercase.

Tokenization and Sequence Creation: The text is tokenized into sequences of words, and sequences are created for training the LSTM.

Model Training: The model is trained on the tokenized sequences to predict the next word in the sequence.

Text Generation: After training, users can interactively input a seed text, and the model will predict the next words.

Save the Model: The trained model and tokenizer are saved for future use.

## 4. Example Commands

Run the main.py file to train the model and generate sentences:

python main.py

Enter seed text and the number of words you'd like to generate.

## 5. User Interface
![Screenshot 2024-09-28 172924](https://github.com/user-attachments/assets/ebe87462-8baf-48ab-b251-c7061fde8d28)
![Screenshot 2024-09-28 173030](https://github.com/user-attachments/assets/b79ae5d1-5995-48bb-91d9-57ac663c5e50)
![Screenshot 2024-09-28 172948](https://github.com/user-attachments/assets/f0bd095a-0c46-4dd4-8438-4d16949d5e66)


# Conclusion
This project successfully demonstrates how an LSTM model can be trained on historical literary texts like Shakespeare's plays to provide dynamic word suggestions. The interactive nature of the interface allows for fun exploration of text generation based on classical literature.
