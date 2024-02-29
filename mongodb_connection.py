from pymongo import MongoClient


def get_database():
    CONNECTION_STRING = "YOUR MONGODB CONNECTION STRING"

    client = MongoClient(CONNECTION_STRING)

    return client['MovieData']


if __name__ == "__main__":
    # Get the database
    dbname = get_database()
