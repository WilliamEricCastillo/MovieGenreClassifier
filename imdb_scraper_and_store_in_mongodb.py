import requests
from bs4 import BeautifulSoup
from mongodb_connection import get_database

dbname = get_database()

# Update table name accordingly ex "top_100_action"
collection_name = dbname["TABLE NAME"]

# URL of the IMDb list ex https://www.imdb.com/list/ls021344401/
url = "IMDb URL"

response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

# Find all <p> elements with class "" update accordingly to match html 
plot_elements = soup.find_all("p", class_="")

plot_summaries = [plot.text.strip() for plot in plot_elements]

for plots in plot_summaries:
    if "/" not in plots:
        item = {
            "plot": plots,
            # Update with genre name
            "genre": "GENRE NAME",
        }
        collection_name.insert_one(item)

print("Done")
