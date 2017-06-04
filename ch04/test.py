import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import gensim

t = '''
        The book covers 24 chapters altogether. It starts with the behavioral perspective of the ‘human cognition’ and covers in detail the tools and techniques required for its intelligent realization on machines. The classical chapters on search, symbolic logic, planning and machine learning have been covered in sufficient details, including the latest research in the subject. The modern aspects of soft computing have been introduced from the first principles and discussed in a semi-informal manner, so that a beginner of the subject is able to grasp it with minimal effort. Besides soft computing, the other leading aspects of current AI research covered in the book include non- monotonic and spatio-temporal reasoning, knowledge acquisition, verification, validation and maintenance issues, realization of cognition on machines and the architecture of AI machines. The book ends with two case studies: one on ‘criminal investigation’ and the other on ‘navigational planning of robots,’ where the main emphasis is given on the realization of intelligent systems using the methodologies covered in the book.
        The book is unique for its diversity in contents, clarity and precision of presentation and the overall completeness of its chapters. It requires no mathematical prerequisites beyond the high school algebra and elementary differential calculus; however, a mathematical maturity is required to follow the logical concepts presented therein. An elementary background of data structure and a high level programming language like Pascal or C is helpful to understand the book. The book, thus, though meant for two semester courses of computer science, will be equally useful to readers of other engineering disciplines and psychology as well as for its diverse contents, clear presentation and minimum prerequisite requirements.
    '''

doc_set = []
doc_set.append(t)

# for folderName, subfolders, filenames in os.walk('/Users/alexeydemyanchuk/Documents/Artificial Intelligence/English/Computer vision'):
#
#     for filename in filenames:
#         if filename.endswith('.pdf'):
#             doc_set.append(filename)

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

# print(len(texts), texts)

# Load the preprocessed corpus (id2word & mm):
id2word = gensim.corpora.Dictionary.load_from_text(
    'data_wordids.txt.bz2')
# mm = gensim.corpora.MmCorpus('data_tfidf.mm')

doc_bow = [id2word.doc2bow(text) for text in texts]


model = gensim.models.ldamodel.LdaModel.load('wiki_lda.pkl')


a = list(sorted(model[doc_bow[0]]))
print(texts[0])
print(a)
print(model.print_topic(a[0][0]))
print(model.print_topic(a[-1][0]))