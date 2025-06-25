from sklearn.tree import DecisionTreeClassifier
import numpy as np

class MyModel:
  def __init__(self, min_samples_split=2, max_depth=5):
    self.clf = DecisionTreeClassifier(criterion="gini", min_samples_split=min_samples_split, max_depth=max_depth)

  def my_fit(self, words):
    # Create features (binary representation of bigram presence)
    features = []
    bigram_set = set(all_bigrams(words))  # Function to generate all possible bigrams
    for word in words:
      feature = [1 if bigram in word else 0 for bigram in bigram_set]
      features.append(feature)

    # Extract target labels (words)
    targets = words

    # Train the decision tree model
    self.clf.fit(features, targets)
    return self

  def my_predict(self, bigrams):
    # Create feature for the input bigrams
    feature = [1 if bigram in bigrams else 0 for bigram in set(bigrams)]
    feature = np.array([feature])

    # Predict the class (word)
    prediction = self.clf.predict(feature)[0]

    # Return a list with the prediction (up to 5 guesses)
    return [prediction]

# Function to generate all possible bigrams from a list of words
def all_bigrams(word):
  return [word[i:i+2] for i in range(len(word) - 1)]

# Example usage (assuming you have a dictionary 'words')
with open("base.txt", 'r') as f:
        words = f.read().splitlines()  # Use splitlines to handle newline characters correctly

model = MyModel()
model.my_fit(words)
bigrams = all_bigrams('optional')
print(bigrams)
# guesses = model.my_predict(bigrams)
