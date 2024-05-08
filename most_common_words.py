import string

#filename = "25k_with_ablation_noE5E6E2S1/train_extracted.txt"
filename = "25k_no_ablation_baseline/train_extracted.txt"

with open(filename, 'r') as f:
    sentences = [line for line in f.read().splitlines() if line.strip()]

# do some preprocessing on the words
for i, s in enumerate(sentences):
    sentences[i] = s.translate(str.maketrans('', '', string.punctuation)).lower()

word_counts = dict()

for s in sentences:
    for w in s.split(' '):
        word_counts[w] = word_counts.get(w,0) + 1

total_word_count = sum([len(s) for s in sentences])
most_frequent_words = sorted(list(word_counts.items()), key=lambda x: -word_counts[x[0]])
for w,c in most_frequent_words[:50]:
    print(w, c / total_word_count)
