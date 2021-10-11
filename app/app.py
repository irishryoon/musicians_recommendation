import numpy as np
import pickle 
import streamlit as st
from gensim.models import KeyedVectors

st.set_page_config(page_title = "musicians", page_icon = ":musical_note")
@st.cache(show_spinner = True, allow_output_mutation = True)
def load_artists():
    with open('artists.pickle', 'rb') as f:
        artists_ID = pickle.load(f)
    
    with open('artists_ID_80000.pkl', 'rb') as f:
        ID_artists = pickle.load(f)

    artists = list(artists_ID.keys())  
    artists.sort()

    return artists_ID, ID_artists, artists

@st.cache(show_spinner = True, allow_output_mutation=True)
def load_embedding():
    # load selected model
    wv = KeyedVectors.load("word2vec.wordvectors", mmap = 'r')
    return wv

def recommend_artists(favorites, wv, n):
    # return recommended artists based on the average vector of the favorite artists
    """
    --- input ---
    favorites: (list) of favorite artists IDs
    wv: KeyedVectors
    n: (int) number of artists to recommend
    --- output ---
    recommend: (list) of artists ID to recommend
    """
    
    v = np.zeros((wv.vector_size, ))
    for artist in favorites:
        v += wv[artist]
    v = v/len(favorites)
    
    rec = wv.similar_by_vector(v, topn = n + len(favorites)) 
    rec = [item for item in rec if item[0] not in favorites]
    return rec

def main():

    st.title("Classical Musicians Recommender")
    st.subheader("Select your favorite performers")
    
    artists_ID, ID_artists, artists = load_artists()
    wv = load_embedding()

    selected_artists = st.multiselect('Type and select your favorite performers.', artists)
    button = st.button("Click here to receive recommendations")

    # button indicating that user finished selected.
    if button:
        # get ID's of favorite artists
        selected_ID = [artists_ID[item] for item in selected_artists]

        rec = recommend_artists(selected_ID, wv, 20)
        rec = [ID_artists[item[0]] for item in rec]
        st.subheader("Recommended artists")
        for item in rec:
            st.write(item)

    st.write("Creator: [Iris Yoon](https://irisyoon.org/)")
    st.write("Code: [github.com/irishryoon/musicians](https://github.com/irishryoon/musicians.git)")
    st.write("Blog post: [medium](https://medium.com/@irishryoon/classical-musicians-recommender-22ee176daee8)")
if __name__ == '__main__':
    main()