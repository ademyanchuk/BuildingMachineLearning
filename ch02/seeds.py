from load import load_dataset
import numpy as np

feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficien',
    'length of kernel groove',]

features, labels = load_dataset('seeds')

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)

from sklearn.cross_validation import KFold

kf = KFold(len(features), n_folds=5, shuffle=True)
# `means` will be a list of mean accuracies (one entry per fold)
means = []
for training,testing in kf:
    # We fit a model for this fold, then apply it to the
    # testing data with `predict`:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])

    # np.mean on an array of booleans returns fraction
    # of correct decisions for this fold:
    curmean = np.mean(prediction == labels[testing])
    means.append(curmean)

print("Mean accuracy: {:.1%}".format(np.mean(means)))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
classifier = KNeighborsClassifier(n_neighbors=1)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])
