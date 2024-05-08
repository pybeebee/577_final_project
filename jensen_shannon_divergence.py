import scipy
import string

filename1 = "5k_all_instruction_types_extracted.txt"
filename2 = "25k_no_ablation_baseline/train_extracted.txt"
size = 100 # parameter for upper bound on number of most frequent words (relative to `filename1`)

##### CODE STARTS HERE #####

def get_freqs(filename):
    '''Returns list of dictionary items for most common tokens and their frequencies
    Key = token, Value = probability of token (length = vocabulary size)'''

    with open(filename, 'r') as f:
        sentences = [line for line in f.read().splitlines() if line.strip()]

    # do some preprocessing on the words
    for i, s in enumerate(sentences):
        sentences[i] = s.translate(str.maketrans('', '', string.punctuation)).lower()

    word_counts = dict()
    total_word_count = sum([len(s) for s in sentences])

    for s in sentences:
        for w in s.split(' '):
            word_counts[w] = word_counts.get(w,0) + 1

    for w in word_counts:
        word_counts[w] /= total_word_count

    most_frequent_words = sorted(list(word_counts.items()), key=lambda x: -word_counts[x[0]])
    return most_frequent_words

all_freqs1 = get_freqs(filename1)
all_freqs2 = get_freqs(filename2)

vocab = [i[0] for i in all_freqs1[:size]]
top_freqs1 = [i[1] for i in all_freqs1[:size]]

top_freqs2 = list()
for w in vocab:
    freq = next((x for x in all_freqs2 if x[0] == w), None)[1]
    top_freqs2.append(freq)

print(scipy.spatial.distance.jensenshannon(top_freqs1, top_freqs2))
