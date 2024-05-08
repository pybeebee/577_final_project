import string

filename = "25k_no_ablation_baseline/train_extracted.txt"

with open(filename, 'r') as f:
    sentences = [line for line in f.read().splitlines() if line.strip()]

# do some preprocessing on the words
for i, s in enumerate(sentences):
    sentences[i] = s.translate(str.maketrans('', '', string.punctuation)).lower()

word_set = set().union(*[set(s.split(' ')) for s in sentences])
print(len(word_set) / sum([len(sentences) for s in sentences]))
