from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
print(vectorizer)

content = ["How to format my hard disk", " Hard disk format problems"]
X = vectorizer.fit_transform(content)
vectorizer.get_feature_names()

print(X.toarray().transpose())

# example of related posts
import os
import sys

import scipy as sp

from sklearn.feature_extraction.text import CountVectorizer

from utils import DATA_DIR

# extracting posts from folder
TOY_DIR = os.path.join(DATA_DIR, "toy")
posts = [open(os.path.join(TOY_DIR, f)).read() for f in os.listdir(TOY_DIR)]

new_post = "imaging databases"

# set vectorizer
vectorizer = CountVectorizer(min_df=1)
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

best_doc = None
best_dist = sys.maxsize
best_i = None
