# Get user input
word = input("Enter word: ")

# Print the word
print(word)

# Count the number of characters in the word
count = len(word)
print(count)

# Initialize variables for creating bigrams
i = 0
j = 1
bigrams = []

# Generate bigrams
while j < count:
    bigram = word[i:j+1]
    i += 1
    j += 1
    bigrams.append(bigram)

# Print the list of bigrams
print(bigrams)

# Sort the list of bigrams
bigrams.sort()

# Print the sorted list of bigrams
print(bigrams)

# Remove duplicates by converting the list to a set and back to a list
unique_bigrams = list(set(bigrams))

# Sort the unique bigrams again (to ensure order after deduplication)
unique_bigrams.sort()

# Print the first 5 unique bigrams
print(unique_bigrams[:5])
print(tuple(unique_bigrams))
