import sklearn.datasets
import scipy as sp


# all data from sklearn dataset 20 newsgroups
all_data = sklearn.datasets.fetch_20newsgroups(subset="all")
print(len(all_data.filenames))
print(all_data.target_names)

groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.fetch_20newsgroups(subset="train", categories=groups)
print("Number of training posts in tech groups:", len(train_data.filenames))
test_data = sklearn.datasets.fetch_20newsgroups(subset="test", categories=groups)
print("Number of testing posts in tech groups:", len(test_data.filenames))

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                                    stop_words="english", decode_error="ignore"
                                    )
vectorized = vectorizer.fit_transform(train_data.data)
num_samples, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))
# samples: 3529, #features: 4712

num_clusters = 50
from sklearn.cluster import KMeans
km = KMeans(n_clusters=num_clusters, init="random", n_init=1,
            verbose=1, random_state=3
            )
km.fit(vectorized)

print("km.labels_=%s" % km.labels_)
# km.labels_=[ 6 34 22 ...,  2 21 26]

print("km.labels_.shape=%s" % km.labels_.shape)
# km.labels_.shape=3529

new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
similar_indices = (km.labels_==new_post_label).nonzero()[0]

similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, train_data.data[i]))

similar = sorted(similar)
print("Count similar: %i" % len(similar))

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 10)]
show_at_3 = similar[int(len(similar) / 2)]

print("=== #1 ===")
print(show_at_1)
print()

print("=== #2 ===")
print(show_at_2)
print()

print("=== #3 ===")
print(show_at_3)

post_group = zip(train_data.data, train_data.target)
# Create a list of tuples that can be sorted by
# the length of the posts
all = [(len(post[0]), post[0], train_data.target_names[post[1]])
       for post in post_group]
graphics = sorted([post for post in all if post[2] == 'comp.graphics'])
print(graphics[5])

noise_post = graphics[5][1]

analyzer = vectorizer.build_analyzer()
print(list(analyzer(noise_post)))

useful = set(analyzer(noise_post)).intersection(vectorizer.get_feature_names())
print(sorted(useful))
