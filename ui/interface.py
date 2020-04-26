import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from utils import transform


st.title('ACCESS UI test')


user_input = st.text_area("Input text")
user_LengthRatioProcessor = st.slider("LengthRatioProcessor", 0., 1., 0.95, step=0.05)
user_LevenshteinPreprocessor = st.slider("LevenshteinPreprocessor", 0., 1., 0.75, step=0.05)
user_WordRankRatioPreprocessor = st.slider("WordRankRatioPreprocessor", 0., 1., 0.75, step=0.05)
user_SentencePiecePreprocessor = st.slider("SentencePiecePreprocessor", 5000, 30000, 10000, step=5000)

@st.cache
def transform_input(input_text, LengthRatioProcessor, LevenshteinPreprocessor, WordRankRatioPreprocessor, SentencePiecePreprocessor):
	print("Computing ", LengthRatioProcessor)
	output_text = transform.transform(input_text, LengthRatioProcessor, LevenshteinPreprocessor, WordRankRatioPreprocessor, SentencePiecePreprocessor) 
	return output_text

transformed_input = transform_input(user_input, user_LengthRatioProcessor, user_LevenshteinPreprocessor, user_WordRankRatioPreprocessor, user_SentencePiecePreprocessor)

'''
### Transformed input
'''
st.write(transformed_input)

