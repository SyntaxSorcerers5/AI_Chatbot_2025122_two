import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from rapidfuzz import fuzz, process
from textblob import TextBlob
from nltk.stem.porter import PorterStemmer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer()

def stem(word):
    """Stem a word to its root form."""
    return stemmer.stem(word.lower())

def tokenize(sentence):
    """Split a sentence into an array of words."""
    return sentence.split()

def bag_of_words(tokenized_sentence, all_words):
    """Convert a sentence into a bag-of-words representation."""
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    import numpy as np
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Set of English stopwords
stop_words = set(stopwords.words("english"))

def tokenize(sentence):
    """Tokenize a sentence into words and handle punctuation properly."""
    # Use NLTK's word_tokenize for initial tokenization
    tokens = nltk.word_tokenize(sentence)
    # Remove punctuation tokens
    tokens = [word for word in tokens if word.isalnum()]
    return tokens

def lemmatize(word):
    """Lemmatize a word to its base form."""
    return lemmatizer.lemmatize(word.lower())

def correct_spelling(sentence):
    """Correct spelling in a sentence using TextBlob."""
    return str(TextBlob(sentence).correct())

def preprocess_input(user_input):
    """
    Preprocess user input:
    - Correct spelling
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    # Correct spelling
    corrected_input = correct_spelling(user_input)
    # Tokenize corrected input
    tokens = tokenize(corrected_input)
    # Remove stopwords and lemmatize the remaining tokens
    lemmatized_tokens = [lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return " ".join(lemmatized_tokens)

def fuzzy_match_with_synonyms(input_phrase, keys, threshold=80):
    """
    Match an input phrase to keys with fuzzy matching and synonym handling.
    """
    # Preprocess input phrase
    processed_input = preprocess_input(input_phrase)

    # Add synonyms (you can expand this dictionary based on your needs)
    synonyms = {
        "cut": ["laceration", "gash", "wound"],
        "burn": ["scald", "blister"],
        "fainting": ["syncope", "blackout"],
    }

    # Expand keys with synonyms
    extended_keys = set(keys)
    for key, syn_list in synonyms.items():
        if key in keys:
            extended_keys.update(syn_list)

    # Perform fuzzy matching
    best_match = process.extractOne(processed_input, extended_keys, scorer=fuzz.ratio)
    if best_match and best_match[1] >= threshold:
        return best_match[0]

    return None