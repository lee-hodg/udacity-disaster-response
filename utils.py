import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def tokenize(text: str):
    """
    NLP pre-processing. First tokenize the message text using NLTK word_tokenize.
    Next use the WordNetLemmatizer and lower-case/strip the lemmatized tokens

    :param text: the text document (message)
    :return: a list of cleaned tokens representing the message
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    # Replace urls
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Normalize and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Lemmatize and remove stop words
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # clean_tokens = [lemmatizer.lemmatize(tok, pos='v').lower().strip() for tok in clean_tokens]

    return clean_tokens
