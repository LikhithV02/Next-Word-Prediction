import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load the saved model
model = load_model('Final_model.h5')

data = pd.read_csv('Movie_Titles.csv')
data['movie_title'] = data['movie_title'].apply(lambda x: x.replace(u'\xa0',u' '))
data['movie_title'] = data['movie_title'].apply(lambda x: x.replace('\u200a',' '))
tokenizer = Tokenizer(oov_token='<oov>') # For those words which are not found in word_index
tokenizer.fit_on_texts(data['movie_title'])
total_words = len(tokenizer.word_index) + 1
input_sequences = []
for line in data['movie_title']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    print(token_list)
    
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text

with gr.Blocks() as demo:
    gr.HTML("<h1><center>Next Word Prediction: Unveiling the Future, One Word at a Time</center></h1>")
    txt = gr.Textbox(label="Your initial word", lines=1)
    slider = gr.Slider(minimum=0, maximum=10, step=1, value=1, label="Number of words")
    txt_3 = gr.Textbox(value="", label="Output")
    btn = gr.Button(value="Submit")
    btn.click(generate_text, inputs=[txt, slider], outputs=[txt_3])


if __name__ == "__main__":
    demo.launch(share=True)