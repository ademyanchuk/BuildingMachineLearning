from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from gensim import corpora, models
import os
import PyPDF2

doc_set = []

for folderName, subfolders, filenames in os.walk('/Users/alexeydemyanchuk/Documents/Artificial Intelligence/English/Computer vision'):

    for filename in filenames:
        if filename.endswith('.pdf'):
            pathToPdf = os.path.join(folderName, filename)
            pdfFileObj = open(pathToPdf, 'rb')
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
            pdfSize = pdfReader.numPages
            page = pdfSize // 2
            pageObj = pdfReader.getPage(page)
            try:
                text = pageObj.extractText()
                if len(pageObj.extractText()) > 500:
                    doc_set.append(text)
            except:
                pass
            pdfFileObj.close()





tokenizer = RegexpTokenizer(r'\w+')


# create English stop words list
en_stop = stopwords.words('english')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
#
texts = []
for doc in doc_set:
    # clean and tokenize document string
    raw = doc.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [token for token in tokens if token not in en_stop]

    words_only_tokens = [token for token in stopped_tokens if wordnet.synsets(token)]

    stemmed_tokens = [p_stemmer.stem(token) for token in words_only_tokens]

    texts.append(stemmed_tokens)
print(len(texts), texts)

# # turn our tokenized documents into a id <-> term dictionary
# dictionary = corpora.Dictionary(texts)
# print(dictionary)
#
# # convert tokenized documents into a document-term matrix
# corpus = [dictionary.doc2bow(text) for text in texts]
# print(corpus)

# ldamodel = models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=200)

# print(ldamodel.print_topics(num_topics=5, num_words=4))