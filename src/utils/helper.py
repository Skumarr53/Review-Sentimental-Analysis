import re
import numpy as np
import spacy
from bs4 import BeautifulSoup
from src.logger.logger import logging
from src.exception.exception import DetailedError


def decontracted(phrase):
    #order should not change
    try:
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    except Exception as e:
        logging.exception(f"Error in decontracted: {e}")
        raise DetailedError(e)

def text_cleaning(text):
    try:
        text = re.sub(r"http\S+", "", text)
        text = BeautifulSoup(text, 'lxml').get_text()
        text = decontracted(text)
        text = re.sub("\S*\d\S*", "", text).strip()
        text = re.sub('[^A-Za-z]+', ' ', text)
        # https://gist.github.com/sebleier/554280
        text = text.lower()
        return text
    except Exception as e:
        logging.exception(f"Error in text_cleaning: {e}")
        raise DetailedError(e)
    
def df_remove_nans(data):
    return data[~(data.Summary.isna() | data.Text.isna())]