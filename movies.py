import pandas as pd
import numpy as np 
import streamlit as st
import pickle 
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



final = pd.read_pickle('final.pkl')
 

porter = PorterStemmer()

final['movie_tags'] = final['movie_tags'].apply(porter.stem)
 
countVec = CountVectorizer(max_features=10000, stop_words='english')

Vectors = countVec.fit_transform(final['movie_tags']).toarray()

similarity = cosine_similarity(Vectors)

def recommend_mmovies(Movie):
    movie_index = final[final['title'] == Movie].index[0]
    distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        recommended_movie_names.append(final.iloc[i[0]].title)

    return recommended_movie_names       

st.title('Movie Recommedation System')
st.subheader('By Krishna Bakshi - www.linkedin.com/in/krishnabakshi')
movie_list = final['title'].values
selected_movie = st.selectbox(
    "Select or Type a movie.",
    movie_list)

if st.button('Click to see recommendations!'):
    movienames = recommend_mmovies(selected_movie)
    for i in movienames:
        st.write(i)    
