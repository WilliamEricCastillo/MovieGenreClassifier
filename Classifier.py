from nltk.tokenize import word_tokenize
from nltk import pos_tag


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
