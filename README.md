# Movie Genre Classifier

This Python program utilizes APIs to retrieve movie data and then classifies movie genres based on their plots using natural language processing (NLP) techniques and machine learning.

## APIs Used
- [MoviesVerse API](https://rapidapi.com/Murad123/api/moviesverse1/): This API provides access to a collection of movies, including their titles.
- [OMDb API](https://www.omdbapi.com/): This API is used to retrieve detailed information about movies, including their genres and plots.

## Dependencies
- `requests`: For making HTTP requests to the APIs.
- `nltk`: Natural Language Toolkit for tokenization and part-of-speech tagging.
- `pickle`: For serializing and deserializing Python objects.
- `sklearn`: Scikit-learn library for machine learning tasks.

## Usage
1. Run the script.
2. Enter the plot of a movie when prompted.
3. The program will predict the genre of the movie based on the provided plot.

## How It Works
1. Retrieves a list of top movies from the MoviesVerse API.
2. Fetches detailed movie data from the OMDb API for each movie.
3. Tokenizes and tags the plot of each movie to extract relevant keywords.
4. Trains a Support Vector Machine (SVM) model using TF-IDF vectorization of the extracted keywords.
5. Persists the trained model and TF-IDF vectorizer for future use.
6. Accepts user input in the form of a movie plot.
7. Preprocesses the user input and predicts the genre using the trained model.
8. Outputs the predicted genre of the movie.

## Files
- `movie_genre_model.pkl`: Serialized SVM model for genre classification.
- `tfidf_vectorizer.pkl`: Serialized TF-IDF vectorizer.
- `README.md`: This documentation.

## Setup
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the script.

## Contributing
Contributions are welcome! Feel free to submit pull requests or open issues.
