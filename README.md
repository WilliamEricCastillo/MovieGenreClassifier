# Movie Genre Classifier

This repository contains Python scripts to scrape IMDb for movie plot summaries and store them in a MongoDB database, as well as to classify movie genres based on plot summaries using a Naive Bayes classifier.

## Requirements

To run the scripts in this repository, you need the following:

- Python 3.x
- `pymongo` library for MongoDB interaction
- `requests` library for making HTTP requests
- `beautifulsoup4` library for web scraping
- `nltk` library for natural language processing tasks
- `scikit-learn` library for machine learning tasks
- `matplotlib` library for data visualization
- `seaborn` library for enhanced data visualization

## Setup

1. Install the required Python libraries using pip:

    ```
    pip install pymongo requests beautifulsoup4 nltk scikit-learn matplotlib seaborn
    ```

2. Ensure you have a MongoDB instance running and obtain the connection string.

## Usage

### 1. MongoDB Connection

Filename: `mongodb_connections.py`

This script establishes a connection to your MongoDB database. You need to replace `"YOUR MONGODB CONNECTION STRING"` with your actual MongoDB connection string.

### 2. IMDb Scraper

Filename: `imdb_scraper_and_store_in_mongodb.py`

This script scrapes IMDb for movie plot summaries, stores them in the MongoDB database, and associates them with respective genres.

To use this script:
- Update the `TABLE NAME` and `IMDb URL` variables accordingly.
- Run the script.

### 3. Movie Genre Classifier

Filename: `Classifier.py`

This script performs the following tasks:
- Tokenizes and processes plot summaries, extracting nouns and adjectives.
- Populates a dataset with keywords extracted from plot summaries stored in MongoDB.
- Trains a Multinomial Naive Bayes classifier on the dataset.
- Predicts the genre of a given plot summary based on user input.
- Evaluates the model's performance using test data.

To use this script:
- Ensure MongoDB is populated with movie plot summaries.
- Run the script in `Classifier.py`.

## Limitations
- The IMDb scraper relies on the structure of the IMDb website. Changes to the website structure may break the scraper.
- The accuracy of the genre classifier may be limited by the quality and diversity of the training data.

## Results

The classifier achieved a test accuracy of 28% on a diverse dataset of movie genres.

### Contributors

- William Castillo 
