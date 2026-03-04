import streamlit as st 
import numpy as np 
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.utils  import pad_sequences 

### load the mode 

model=load_model('next_word_lstm.keras')

with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)


# function to predict 
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len)]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predict_word_index = np.argmax(predicted , axis=1)
    for word ,index in tokenizer.word_index.items():
        if index == predict_word_index:
            return word 
    return none 


## UI 
st.title("Next word prediction")
input_text = st.text_input("Enter the sequence of words ", " what is life if not")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(input_text + " " + next_word)

