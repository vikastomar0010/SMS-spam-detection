import nltk as nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st
import pickle
import string

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.title("sms spam classifier")

input_sms = st.text_input("enter your message")

transformed_sms = transform_text(input_sms)
# Use transform() method with parentheses to transform the text
vector_input = tfidf.transform([transformed_sms])
result = model.predict(vector_input)[0] # Access the prediction result
if result == 1:
    st.header("spam")
else:
    st.header("not spam")