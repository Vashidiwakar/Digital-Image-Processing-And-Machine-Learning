import itertools
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Function to generate bigrams from a word
def generate_bigrams(word):
    return [word[i:i+2] for i in range(len(word) - 1)]

# Fit function to train the model
def my_fit(dictionary):
    # Create a list of unique bigrams
    bigram_set = set()
    for word in dictionary:
        bigrams = generate_bigrams(word)
        bigram_set.update(bigrams)

    bigram_list = sorted(list(bigram_set))

    # Create the feature matrix
    X = []
    y = []

    for word in dictionary:
        bigrams = generate_bigrams(word)
        features = [1 if bigram in bigrams else 0 for bigram in bigram_list]
        X.append(features)
        y.append(word)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Train a decision tree classifier
    clf = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5)
    clf.fit(X, y)

    return clf, bigram_list

# Helper function to check if all bigrams are present in order and consecutively
def bigrams_in_order_and_consecutive(word, bigrams):
    word_bigrams = generate_bigrams(word)
    
    # Check for consecutive occurrence
    index = 0
    for bigram in bigrams:
        try:
            index = word_bigrams.index(bigram, index)
        except ValueError:
            return False
        index += 1
    return True

# Predict function to guess words based on bigrams
def my_predict(model, bigram_tuple):
    clf, bigram_list = model

    # Create the feature vector for the input bigrams
    features = [1 if bigram in bigram_tuple else 0 for bigram in bigram_list]
    features = np.array(features).reshape(1, -1)

    # Predict probabilities
    probabilities = clf.predict_proba(features)[0]

    # Get all possible guesses
    all_guesses = clf.classes_
    
    # Filter guesses to include only those that match the criteria
    filtered_guesses = [word for word in all_guesses if bigrams_in_order_and_consecutive(word, bigram_tuple)]

    # Get probabilities for filtered guesses
    filtered_probabilities = [probabilities[np.where(all_guesses == word)[0][0]] for word in filtered_guesses]
    
    # Get the top 5 guesses
    if len(filtered_probabilities) > 0:
        top_indices = np.argsort(filtered_probabilities)[-5:][::-1]
        guesses = [filtered_guesses[i] for i in top_indices]
    else:
        guesses = []

    return list(guesses)

# Example usage
if __name__ == "__main__":
    with open("base.txt", 'r') as f:
        dictionary = f.read().splitlines()  # Use splitlines to handle newline characters correctly

    model = my_fit(dictionary)
    bigrams = generate_bigrams("optional")
    print("Bigrams:", bigrams)
    unique_bigrams = list(set(bigrams))

    # Sort the unique bigrams again (to ensure order after deduplication)
    unique_bigrams.sort()

    # Create a tuple of the first 5 unique bigrams
    bigram_tuple = tuple(unique_bigrams[:5])
    print("Bigram Tuple:", bigram_tuple)
    guesses = my_predict(model, bigram_tuple)
    print("Guesses:", guesses)
