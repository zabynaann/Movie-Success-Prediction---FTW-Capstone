import pandas as pd
import streamlit as st
import joblib

import numpy as np

import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

import gensim
from gensim import corpora
from gensim.models import CoherenceModel

import streamlit as st

nltk.download('stopwords')
nltk.download('wordnet')

# Header
### Set Title
st.title('_Kita mo KITA mo!_')
st.header("A Movie Success Predictor from High Gross Revenue")
st.write("""From the movie aspects selected, find out if your movie will become a box office hit!""")

# Read Data
data=pd.read_csv('data/FINAL_NA_TALAGA.csv')

###############################################################################
# DATA CLEANING

data['FLAG_TOP_GROSS'] = np.where(data['GROSS'] >= 2000000, 1, 0)

# Convert RELEASE DATE Data Type to "DATE"
data['RELEASE_DATE'] = pd.to_datetime(data['RELEASE_DATE'], errors='coerce')

# New Column for Release Month from Release Date
data['RELEASE_MONTH'] = pd.DatetimeIndex(data['RELEASE_DATE']).month

#change numerical months into words
import calendar
data['RELEASE_MONTH'] = data['RELEASE_MONTH'].apply(lambda x: calendar.month_name[x])

# Fill null values
data['MOVIE_CERTIFICATE'] = data['MOVIE_CERTIFICATE'].fillna('Not Rated')
data['RUNTIME'] = data['RUNTIME'].fillna(data.RUNTIME.mean())

# Split Countries and concatenate it with the data
data = pd.concat([data, data.COUNTRIES.str.split(',',expand=True)],1)

# Rename Columns
data = data.rename(columns={0: 'COUNTRY_1', 1: 'COUNTRY_2', 2:'COUNTRY_3', 3:'COUNTRY_4', 4:'COUNTRY_5', 5:'COUNTRY_6', 6:'COUNTRY_7', 7:'COUNTRY_8', 8:'COUNTRY_9'})

# Drop other Country Columns, retain only Country_1
data = data.drop(columns = ["COUNTRY_2", "COUNTRY_3", "COUNTRY_4", "COUNTRY_5", "COUNTRY_6", "COUNTRY_7", "COUNTRY_8", "COUNTRY_9"])

# Fill Null Countries with USA (Country with majority count)
data['COUNTRY_1'] = data['COUNTRY_1'].fillna('USA')

# Split Languages and concatenate it with the data
data = pd.concat([data, data.LANGUAGES.str.split(',',expand=True)], 1)

# Rename Columns
data = data.rename(columns={0: 'LANGUAGE_1', 1: 'LANGUAGE_2', 2:'LANGUAGE_3', 3:'LANGUAGE_4', 4:'LANGUAGE_5', 5:'LANGUAGE_6', 6:'LANGUAGE_7', 7:'LANGUAGE_8', 8:'LANGUAGE_9', 9:'LANGUAGE_10'})

# Drop other Language Columns, retain only Language_1
data = data.drop(columns = ["LANGUAGE_2", "LANGUAGE_3", "LANGUAGE_4", "LANGUAGE_5", "LANGUAGE_6", "LANGUAGE_7", "LANGUAGE_8", "LANGUAGE_9", "LANGUAGE_10"])

###############################################################################

# Text input
PLOT = st.text_area("Write your storyline.", 'Input text here.')
GENRE = st.text_area("What genre/s do you want?", 'Input text here.')

# Create Sidebar
# Sidebar Description
st.sidebar.header('Movie Aspects')

# Runtime Slider
RUNTIME = st.sidebar.slider('Runtime', 60, 200, 130)

# Movie Certificate
MOVIE_CERTIFICATE_VALUES = pd.Series(data['MOVIE_CERTIFICATE'].unique()).str.strip()
MOVIE_CERTIFICATE_DUMMIES = pd.get_dummies(MOVIE_CERTIFICATE_VALUES)
st.sidebar.subheader('Movie Certificate')
MOVIE_CERTIFICATE_SAMPLE = st.sidebar.selectbox("Select one from the list", MOVIE_CERTIFICATE_VALUES.values.tolist())
MOVIE_CERTIFICATE_SAMPLE_DUMMIES = (MOVIE_CERTIFICATE_DUMMIES.loc[np.where(MOVIE_CERTIFICATE_VALUES.values == MOVIE_CERTIFICATE_SAMPLE)[0]]
                                  .values.tolist()[0])

# Release Month
RELEASE_MONTH_VALUES = pd.Series(data['RELEASE_MONTH'].unique()).str.strip()
RELEASE_MONTH_DUMMIES = pd.get_dummies(RELEASE_MONTH_VALUES)
st.sidebar.subheader('Month')
RELEASE_MONTH_SAMPLE = st.sidebar.selectbox("Select one from the list", RELEASE_MONTH_VALUES.values.tolist())
RELEASE_MONTH_SAMPLE_DUMMIES = (RELEASE_MONTH_DUMMIES.loc[np.where(RELEASE_MONTH_VALUES.values == RELEASE_MONTH_SAMPLE)[0]]
                                  .values.tolist()[0])

# Language
LANGUAGE_VALUES = pd.Series(data['LANGUAGE_1'].unique()).str.strip()
LANGUAGE_DUMMIES = pd.get_dummies(LANGUAGE_VALUES)
st.sidebar.subheader('Language')
LANGUAGE_SAMPLE = st.sidebar.selectbox("Select one from the list", LANGUAGE_VALUES.values.tolist())
LANGUAGE_SAMPLE_DUMMIES = (LANGUAGE_DUMMIES.loc[np.where(LANGUAGE_VALUES.values == LANGUAGE_SAMPLE)[0]]
                                  .values.tolist()[0])

# Country
COUNTRY_VALUES = pd.Series(data['COUNTRY_1'].unique()).str.strip()
COUNTRY_DUMMIES = pd.get_dummies(COUNTRY_VALUES)
st.sidebar.subheader('Country')
COUNTRY_SAMPLE = st.sidebar.selectbox("Select one from the list", COUNTRY_VALUES.values.tolist())
COUNTRY_SAMPLE_DUMMIES = (COUNTRY_DUMMIES.loc[np.where(COUNTRY_VALUES.values == COUNTRY_SAMPLE)[0]]
                                  .values.tolist()[0])

# Function to Clean text
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Actually clean text
text_process = clean(PLOT).split()
text_process1 = clean(GENRE).split()

# Load dictionary
loaded_dictionary = corpora.Dictionary.load('data/dictionary.sav')
loaded_dictionary1 = corpora.Dictionary.load('data/dictionary1.sav')

# Convert list of words to document term array
doc_term = loaded_dictionary.doc2bow(text_process)
doc_term1 = loaded_dictionary1.doc2bow(text_process1)

# Load LDA model
lda_load_model = gensim.models.ldamodel.LdaModel.load("data/lda_model.sav")
lda_load_model1 = gensim.models.ldamodel.LdaModel.load("data/lda_model1.sav")

# Get document topic probabilities
lists = lda_load_model.get_document_topics(doc_term, minimum_probability=0.0)
lists1 = lda_load_model1.get_document_topics(doc_term1, minimum_probability=0.0)

# Convert to array/list
document_topic =  [i[1] for i in lists]
document_topic1 =  [i[1] for i in lists]

# Load regression model
classification_model = joblib.load('rf_model.pkl')

# predict on output document topic
# prediction = regression_model.predict([document_topic + [Age]])[0]
sample_features = [RUNTIME] + document_topic1 + document_topic + MOVIE_CERTIFICATE_SAMPLE_DUMMIES + COUNTRY_SAMPLE_DUMMIES + LANGUAGE_SAMPLE_DUMMIES + RELEASE_MONTH_SAMPLE_DUMMIES
prediction = classification_model.predict([sample_features])[0]

# Write out prediction
st.header("Based on the movie aspects you selected...")

if prediction == 1:
    st.write("...your movie can be a box office hit with gross revenue over Php 100,000,000.00!")
elif prediction == 0:
    st.write("...you may want to tweak your movie aspects a bit more as your gross revenue is below Php 100,000,000.00")