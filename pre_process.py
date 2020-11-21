import json
from tqdm import tqdm

from konlpy.tag import Okt

okt = Okt()

def read_data(filename: str) -> list:
    with open(filename, 'r', encoding='utf8') as f:
        data = [ l.split('\t') for l in f.read().splitlines() ]
        data = data[1:]
    return data

def tokenize(text: str, pbar: tqdm = None) -> list:
    if pbar: pbar.update()
    return [ '/'.join(t) for t in okt.pos(text) ]

train_data= read_data('./data/train.txt')
print('Tokenizing Train Data')
pbar = tqdm(total=len(train_data))
train_res = [ ( tokenize(r[1], pbar), r[2]) for r in train_data ]
with open('train_data.json', 'w', encoding='utf8') as f:
    print('\nSaving Train Data')
    json.dump(train_res, f, ensure_ascii=False)

test_data = read_data('./data/test.txt')
print('Test Train Data')
pbar = tqdm(total=len(test_data))
test_res = [ ( tokenize(r[1], pbar), r[2]) for r in test_data ]
with open('test_data.json', 'w', encoding='utf8') as f:
    print('\nSaving Test Data')
    json.dump(test_res, f, ensure_ascii=False)