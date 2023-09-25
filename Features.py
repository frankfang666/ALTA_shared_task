import nltk
import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, cmudict
import ssl
from LoadData import read_json_objs

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('cmudict')


def remove_noise(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words


def avg_sent_length(text):
    sentences = sent_tokenize(text)
    return np.average([len(sentence) for sentence in sentences])

def avg_word_in_sent(text):
    sentences = sent_tokenize(text)
    return np.average([len(sentence.split()) for sentence in sentences])

def puncuation_count(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = [c for c in text if c in st]
    return len(count) / len(text)

def stopwords_count(text):
    return len([word for word in remove_noise(text) if word in set(stopwords.words('english'))]) / len(remove_noise(text))

def create_feature(text):
    return [avg_sent_length(text), avg_word_in_sent(text), puncuation_count(text), stopwords_count(text)] 


############### fluency features ###############
def flesch_reading_ease(text):
    words = remove_noise(text)
    return 206.835 - 1.015 * (float(len(words)) / ) - 84.6 * ()


# print(word_tokenize("you shouldn't do this"))
# print(set(stopwords.words('english')))
# print(cmudict.dict()['apple'])

# test_text = train_objs[0]['text']
train_objs = read_json_objs('./alta2023_public_data/training.json')

human = [train_obj for train_obj in train_objs if train_obj['label'] == 1]
machine = [train_obj for train_obj in train_objs if train_obj['label'] == 0]

print(create_feature(human[0]['text']))