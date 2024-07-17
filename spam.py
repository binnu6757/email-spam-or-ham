import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer



# streamlit app
st.image(r"new logo.jpg")
st.title("EMAIL SPAM OR HAM CLASSIFIER")

# Loading the trained model 
model = pickle.load(open(r"nb.pkl","rb"))
model1 = pickle.load(open(r"bow.pkl","rb"))

# input text box
email =st.text_area("Please Enter Your Email Text To Classify")
checking = model1.transform([email])
prediction = model.predict(checking)[0]

# Predict Button
if st.button("Predict"):
    if prediction == "spam":
        st.write("This email is SPAM")
        st.image(r"new spam logo.jpg")

    else:
        st.write("This Not A Spam")
        st.image(r"new thumb.jpg")
        








