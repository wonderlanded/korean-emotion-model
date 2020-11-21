import nltk
import json

with open('train_data.json', encoding='utf8') as f:
    train = json.load(f)

tokens = [t for d in train for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

words = [f[0] for f in text.vocab().most_common(10000)]

def term_frequency(doc):
    return [doc.count(word) for word in words]
