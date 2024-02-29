from nltk.tokenize import word_tokenize
from nltk import pos_tag
from mongodb_connection import get_database
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def process_text(text):
    """
    Tokenizes the input text, performs part-of-speech tagging,
    and extracts adjectives and nouns to generate keywords.

    :param text: The input text to be processed.
    :return: A string containing the extracted keywords.
    """
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    keywords_set = {token for token, tag in tagged_tokens if tag.startswith('N') or tag.startswith('J')}
    return ' '.join(keywords_set)


def populate_dataset(dbname, movieBank, dataset):
    """
    Populates the dataset with keywords extracted from the plot summaries
    stored in MongoDB collections corresponding to different movie genres.

    :param dbname: The MongoDB database object.
    :param movieBank: A list of strings representing the names of collections in the database.
    :param dataset: The dataset to be populated with keyword-genre pairs.
    """
    for genres in movieBank:
        collection_name = dbname[genres]
        item_details = collection_name.find()

        for item in item_details:
            keywords = process_text(item["plot"])
            dataset.append((keywords, item["genre"]))


def train_classifier(X, y):
    """
    Trains a Multinomial Naive Bayes classifier on the provided text data.

    :param X: List of text data (plot summaries).
    :param y: List of corresponding labels (genres).
    :return: Tuple containing trained classifier, vectorizer, test features, and test labels.
    """
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    return classifier, vectorizer, X_test, y_test


def predict_genre(plot_summary, classifier, vectorizer):
    """
    Predicts the genre of a given plot summary using a trained classifier and vectorizer.

    :param plot_summary: The plot summary for which the genre is to be predicted.
    :param classifier: The trained classifier model.
    :param vectorizer: The vectorizer used to transform text data.
    :return: The predicted genre of the plot summary.
    """
    plot_summary_processed = process_text(plot_summary)
    plot_summary_vectorized = vectorizer.transform([plot_summary_processed])
    predicted_genre = classifier.predict(plot_summary_vectorized)[0]
    return predicted_genre


def evaluate_model(classifier, X_test, y_test):
    """
    Evaluates the performance of a trained classifier on the test set.

    :param classifier: The trained classifier model.
    :param X_test: The feature matrix of the test set.
    :param y_test: The true labels of the test set.
    :return: The test accuracy score, confusion matrix, and classification report.
    """
    y_pred_test = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    classification_rep = classification_report(y_test, y_pred_test)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classifier.classes_, yticklabels=classifier.classes_)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    return test_accuracy, conf_matrix, classification_rep


def main():
    dbname = get_database()
    movieBank = ["Top_100_Action", "Top_100_Adventure", "Top_100_Comedies", "Top_100_Drama", "Top_100_Fantasy",
                 "Top_100_Horror", "Top_100_Mystery", "Top_100_Romantic", "Top_100_Sci-Fi", "Top_100_Thrillers",
                 "Best_Raunchy_Comedies", "Best_2000s_Comedy"]
    dataset = []

    populate_dataset(dbname, movieBank, dataset)

    # print(dataset)

    X = [text for text, _ in dataset]
    y = [genre for _, genre in dataset]

    classifier, vectorizer, X_test, y_test = train_classifier(X, y)

    plot_summary = input("\nEnter plot summary: ")

    while plot_summary == "":
        print("Plot summary can not be blank")
        plot_summary = input("\nEnter plot summary: ")

    predicted_genre = predict_genre(plot_summary, classifier, vectorizer)
    print("\nPredicted Genre:", predicted_genre)

    test_accuracy, conf_matrix, classification_rep = evaluate_model(classifier, X_test, y_test)

    print("\nTest Accuracy:", test_accuracy)

    print("\nClassification Report:")
    print(classification_rep)


if __name__ == "__main__":
    main()
