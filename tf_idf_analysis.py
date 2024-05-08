#import pandas as pd
#from sklearn.feature_extraction.text import TfidfVectorizer
import math
import string

filename = "25k_with_ablation_noE5E6E2S1/train_extracted.txt"

with open(filename, 'r') as f:
    sentences = [line for line in f.read().splitlines() if line.strip()]

# do some preprocessing on the words
for i, s in enumerate(sentences):
    sentences[i] = s.translate(str.maketrans('', '', string.punctuation)).lower()

word_set = set().union(*[set(s.split(' ')) for s in sentences])
num_docs = len(sentences)
freq_dict = dict()

# intialize word counts
for w in word_set:
    freq_dict[w] = dict()

# update word counts for each document
for s in sentences:
    # every word in vocab gets a freebie
    for w in word_set:
        freq_dict[w][s] = freq_dict[w].get(s,0) + 1
    # but each doc also tallies the words it encounters
    words = s.split(' ')
    for w in words:
        freq_dict[w][s] += 1


# take log_10 of word counts and multiply by inverse document frequencies

for w in word_set:
    occurrences = len([s for s in sentences if w in s.split(' ')])
    idf = math.log(num_docs / occurrences, 10)
    for s in sentences:
        freq_dict[w][s] = math.log(freq_dict[w][s], 10) * idf

# AFTER THIS IS DONE: get the 10 words that are most important from each document
word_counts = dict()

for s in sentences:
    important_words = sorted([w for w in word_set], key=lambda w: -freq_dict[w][s])[:10]
    for w in important_words:
        word_counts[w] = word_counts.get(w,0) + 1

most_important_words = sorted(list(word_counts.items()), key=lambda x: -word_counts[x[0]])
for w,c in most_important_words[:50]:
    print(w,c)
