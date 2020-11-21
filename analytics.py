import json
import nltk

with open('train_data.json') as f:
    train_data = json.load(f)

text = nltk.Text()