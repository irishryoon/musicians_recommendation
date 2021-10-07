FROM python:3.8.3

# create environment
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy files
COPY app.py artists.pickle artists_ID_80000.pkl word2vec.wordvectors word2vec.wordvectors.vectors.npy ./

EXPOSE 8502

#ENTRYPOINT ["streamlit", "run"]
#CMD ["app.py"]
# for Heroku
CMD streamlit run --server.port $PORT app.py