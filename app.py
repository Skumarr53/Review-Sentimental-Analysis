# from flask import Flask, request, render_template, jsonify
import pickle
from src.logger.logger import logging
from src.exception.exception import DetailedError
from src.pipeline.inference_pipeline import PredictionPipeline
import os
import streamlit as st

infer_pipe = PredictionPipeline()

def main():
    st.title("Sentimental analysis")
    
    review_text = st.text_area("Enter Your Review","")

    if st.button("Predict"):
        if review_text == "":
            st.success("No review found")
        else:
            # Dummy prediction logic for demonstration
            prediction = infer_pipe.predict([review_text])
            sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

            st.success(sentiment)


if __name__ == '__main__':
    main()