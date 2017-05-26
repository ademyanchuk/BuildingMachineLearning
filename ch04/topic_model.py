from gensim import corpora, models, matutils
from scipy.spatial import distance
import matplotlib.pyplot as plt

corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')
model = models.ldamodel.LdaModel(
    corpus,
    num_topics=100,
    id2word=corpus.id2word
    )

# doc = corpus.docbyoffset(0)
# topics = model[doc]
# print(topics)

# num_topics_used = [len(model[doc]) for doc in corpus]
# plt.hist(num_topics_used)
# plt.title('Topics distribution')
# plt.xlabel('Nr of topics')
# plt.ylabel('Nr of documents')
# plt.show()

topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)

pairwise = distance.squareform(distance.pdist(topics))

largest = pairwise.max()
for ti in range(len(topics)):
    pairwise[ti,ti] = largest + 1

def closest_to(doc_id):
    return pairwise[doc_id].argmin()


