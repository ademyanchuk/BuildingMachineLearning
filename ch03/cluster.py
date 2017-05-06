from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
print(vectorizer)

content = ["How to format my hard disk", " Hard disk format problems"]
X = vectorizer.fit_transform(content)
vectorizer.get_feature_names()

print(X.toarray().transpose())
