# # API TO GET MOVIE NAMES  https://rapidapi.com/Murad123/api/moviesverse1/
# # API TO PRINT DESCRIPTION https://www.omdbapi.com/

import requests
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pickle
from typing import List, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

url = "https://moviesverse1.p.rapidapi.com/top-250-movies"

headers = {
    "X-RapidAPI-Key": "81f2af2ea4msh5138b7ac94e2c09p19c100jsnb2b9aa094ed4",
    "X-RapidAPI-Host": "moviesverse1.p.rapidapi.com"
}

response = requests.get(url, headers=headers)
data = response.json()

# Replace '[your_api_key]' with your actual API key
#api_key = [your_api_key]

# URL of the OMDB API
omdb_url = f'http://www.omdbapi.com/?apikey={api_key}&'


# Define the structure of the TypedDict
class MovieData(TypedDict):
    Genre: str
    KeyWords: List[str]


# Initialize an empty list to store the movie data
trainingData: List[MovieData] = []

for movie in data['movies']:
    x = movie['title']

    query_param = 't=' + x

    # Sending the request and getting the response
    response = requests.get(omdb_url + query_param)

    # Checking if the request was successful (status code 200)
    if response.status_code == 200:
        movie_data = response.json()

        # Splitting genres if there are multiple ones
        genres = movie_data['Genre'].split(', ')

        # Tokenize and POS tagging
        tokens = word_tokenize(movie_data['Plot'])
        tagged_tokens = pos_tag(tokens)

        # Extract nouns and adjectives
        keyWords = []
        for token, tag in tagged_tokens:
            if tag.startswith('N') or tag.startswith('J'):  # Nouns and adjectives
                keyWords.append(token)

        # Append data to trainingData list with separate entries for each genre
        for genre in genres:
            trainingData.append({
                "Genre": genre,
                "KeyWords": keyWords
            })
    else:
        print(f"Error: {response.status_code}")


# Preprocessing
genres = [movie['Genre'] for movie in trainingData]
keywords = [' '.join(movie['KeyWords']) for movie in trainingData]

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(keywords)

# Model Training
model = SVC(kernel='linear')
model.fit(X, genres)

# Save trained model and vectorizer
with open('movie_genre_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Load trained model and vectorizer
with open('movie_genre_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# User Input
user_plot = input("Enter the plot: ")


# Preprocess user input
def preprocess_plot(plot):
    return tfidf_vectorizer.transform([plot])


# Tokenize the user input
tokens = word_tokenize(user_plot)

# Join the tokens into a single string
user_input = " ".join(tokens)

# Preprocess user input
user_plot_processed = preprocess_plot(user_input)

# Prediction
predicted_genre = model.predict(user_plot_processed)
print()
print("Predicted Genre:", predicted_genre[0])
