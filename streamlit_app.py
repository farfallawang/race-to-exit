#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:35:00 2020

@author: mengdie
"""

import numpy as np
import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import time
import datetime
from datetime import date
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import nltk
from model_helper import *
#import plotly.figure_factory as ff
#import matplotlib.pyplot as plt
#nltk.download('stopwords')
#nltk.download('punkt')


# Load data 
df = pd.read_csv('companies_v3.csv', delimiter=",")
df.description = df.description.astype('str')
tech_subset = ['analytics', 'ecommerce',  'messaging' , 'mobile', 
               'search', 'security', 'social', 'software', 'web']
df_subset = df[df['category_code'].isin(tech_subset)]

file = open('milestone_text.txt', 'r')
text_for_vocab = file.read()
file.close()

''' Get user input '''
st.sidebar.title('Company Metadata')

founding_date = st.sidebar.date_input('Founding Date', datetime.date(2018, 1, 1), max_value = date.today())
years_since_founded = date.today().year - founding_date.year 

unique_categories = list(set(df_subset.category_code)) 
category_code = st.sidebar.selectbox(
    'Industry Sector',
    unique_categories)
milestones = st.sidebar.slider("Number of Milestones Achieved", 0, 10, 3, 1) #90

funding_rounds = st.sidebar.slider("Total Funding Rounds", 1, 10, 5) #14
funding_total_usd = st.sidebar.slider("Total Funding Received in USD (After Log Transformation)", 5, 20, 5, 1)

last_raised_amount_usd = st.sidebar.slider("Last Round Funding in USD (After Log Transformation)", 5, 20, 5, 1)
last_funding_participants_cnt = st.sidebar.slider("Last Round Funding Participants Cnt", 1, 10, 8) #32

unique_funding_round_types = list(set(df_subset.last_funding_round_type))
last_funding_round_type = st.sidebar.selectbox(
    'Last funding round type',
    unique_funding_round_types)

has_invested_other_companies = st.sidebar.checkbox("Has Invested In Other Companies", value = False)
if has_invested_other_companies:
    has_invested_other_companies = 1
else:
    has_invested_other_companies = 0


import base64
file_ = open("Slides pics/flying_ipo.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)

st.markdown('# Welcome! Ready to evaluate your company?')
input_text = st.text_input('Enter the latest news of your company', 'eg. Hired new CEO, Revenue hits million')
#uploaded_file = st.file_uploader("Uploda a CSV file with detailed funding info", type="csv")
#if uploaded_file is not None:
#    data = pd.read_csv(uploaded_file)
#    st.write(data)

''' Load model and saved vectorizer '''
@st.cache(allow_output_mutation=True)
def load_model():
    #model = pickle.load(open('xgboost_nlp_v2.dat', "rb")) 
    model = pickle.load(open('logistic_nlp.dat', "rb"))
    return model

with open('vectorizer_v2.pickle', 'rb') as f:
    vectorizer = pickle.load(f)
with open('vectorizer_2gram_v2.pickle', 'rb') as f:
    vectorizer_2gram = pickle.load(f)
with open('encoder.pickle', 'rb') as f:
    enc = pickle.load(f)


vectors =  vectorizer.transform(input_text.split('\n')) 
tfidf_mat = vectors.toarray()
one_gram_names = vectorizer.get_feature_names()


vectors_2gram =  vectorizer_2gram.transform(input_text.split('\n')) 
tfidf_mat_2gram = vectors_2gram.toarray()
two_gram_names = vectorizer_2gram.get_feature_names()

x_numeric = np.array(pd.DataFrame([years_since_founded, funding_rounds, 
         funding_total_usd, milestones, has_invested_other_companies, 
         last_raised_amount_usd, last_funding_participants_cnt
         ]).T)
x_test_enc = enc.transform(pd.DataFrame([[category_code, last_funding_round_type]])).toarray()


numeric_feats =  ['years_since_founded', 'funding_rounds', 
         'funding_total_usd', 'milestones', 'has_invested_other_companies', 
         'last_raised_amount_usd', 'last_funding_participants_cnt'
         ]
feature_names = numeric_feats + list(enc.get_feature_names()) + one_gram_names + two_gram_names 

x_test = np.concatenate((x_numeric, x_test_enc, tfidf_mat, tfidf_mat_2gram), axis = 1)
x_test_df = pd.DataFrame(x_test, columns = feature_names)


#tmp = ['years_since_founded', 'the number of total funding rounds', 
#         'the total funding received (in usd)', 'the total number of milestones', 'has_invested_other_companies', 
#         'last_raised_amount_usd', 'last_funding_participants_cnt']

st.markdown('### What\'s the probability my company will go ipo/acquired in two years?')
if st.button('Run'):
    #st.text(x_test)
    model = load_model()
    #st.text(feature_names)
    pred_prob = model.predict_proba(x_test_df)[:,1] #model.predict(xgb.DMatrix(x_test))  #, feature_names = feature_names
    str(round(float(pred_prob) * 100, 2)) + '%'

if st.button('View Analysis Report'):
    st.write('### Industry Average')

#    from PIL import Image 
#    image = Image.open('feature_importance.png')
#    st.image(image)
    df_pos = df[df.is_ipo_or_acquired == 1]
    df_sub = df_pos[df_pos.category_code == category_code]
    for feat in numeric_feats:
        avg = df_sub[feat].mean()
        st.text('The average ' + feat + ' of successful exits in your industry is ' + str(round(avg)))
        # Create distplot with custom bin_size
        
    st.write('### Top Three Ways to Boost Exit Probability')  
             
    new_test = get_synthetic_example(feats, vectorizer, vectorizer_2gram, enc)
    
        
        
    st.write('### Text Analysis Results')      
    pred_vocab = []
    if np.sum(tfidf_mat) != 0:
        idx = np.nonzero(tfidf_mat)[1]
        for i in idx:
            #pred_vocab.append(one_gram_names[i])
            st.write( '`', one_gram_names[i], '`', 'is a token frequently appears in sucessful exits with importance score',  round(tfidf_mat[0, i], 2))
    if np.sum(tfidf_mat_2gram) != 0:
        idx = np.nonzero(tfidf_mat_2gram)[1]
        for i in idx: 
            #pred_vocab.append(two_gram_names[i])
    #st.text([pred_vocab[i] for i in range(len(pred_vocab))])
            st.write( '`', two_gram_names[i], '`', 'is a token  frequently appears in sucessful exits with importance score',  round(tfidf_mat_2gram[0, i], 2))
    if np.sum(tfidf_mat) == 0 and np.sum(tfidf_mat_2gram) == 0:
        st.write('The model didn\'t picked up any predictive info from your text input. Try something else!')
    
    #st.write('### Clustering Results')
