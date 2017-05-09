from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1, stop_words="english")
print(vectorizer)

content = ["How to format my hard disk", " Hard disk format problems"]
X = vectorizer.fit_transform(content)
vectorizer.get_feature_names()

print(X.toarray().transpose())





# EXAMPLE OF RELATED POSTS
import os
import sys

import scipy as sp

import nltk.stem

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import DATA_DIR

english_stemmer = nltk.stem.SnowballStemmer('english')

# build a class from CountVectorizer with manually changed analyzer
class StemmedCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer,self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


# extracting posts from folder
TOY_DIR = os.path.join(DATA_DIR, "toy")
posts = [open(os.path.join(TOY_DIR, f)).read() for f in os.listdir(TOY_DIR)]

new_post = "imaging databases"

# set vectorizer with stop_words
vectorizer = StemmedCountVectorizer(min_df=1, stop_words="english")

# to see stop_words uncomment next line
# sorted(vectorizer.get_stop_words())[0:20]

# transform posts
X_train = vectorizer.fit_transform(posts)

num_samples, num_features = X_train.shape
print("#samples: %d, #features: %d" % (num_samples,num_features))

print(vectorizer.get_feature_names())

# transform new post
new_post_vec = vectorizer.transform([new_post])

print(new_post_vec)
print(new_post_vec.toarray())

# calculate the Euclidean distance between the count vectors
def dist_raw(v1, v2):
    delta = v1-v2
    return sp.linalg.norm(delta.toarray())
# normalized distance
def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())

    delta = v1_normalized - v2_normalized

    return sp.linalg.norm(delta.toarray())




best_doc = None
best_dist = sys.maxsize
best_i = None

for i in range(0, num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec, new_post_vec)

    print("=== Post %i with dist=%.2f: %s" % (i, d, post))

    if d < best_dist:
        best_dist = d
        best_i = i

print("Best post is %i with dist=%.2f" % (best_i, best_dist))
