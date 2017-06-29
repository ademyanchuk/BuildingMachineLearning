import nltk.data
from nltk.tokenize import TreebankWordTokenizer

document = "Hello World. It's good to see you. Thanks for buying this book."

sent_tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
tokenized_sents = sent_tokenizer.tokenize(document)

word_tokenizer = TreebankWordTokenizer()
tokenized_words = []
for sent in tokenized_sents:
    tokenized_words.append(word_tokenizer.tokenize(sent))

print(tokenized_words)