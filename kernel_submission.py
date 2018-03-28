import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from svm import SVM
from KLR import KLR


def _get_data(filename, header=None, index_col=None):
    if header is None:
        return pd.read_csv(filename, header=header).values.tolist()
    else:
        return pd.read_csv(filename).values.tolist()


# test using random split (20% of the data used as validation set)
def test_random_split(clf):
    scores = []
    for idx_data in range(3):
        X = train_dna[idx_data]
        y = label_dna[idx_data]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, stratify=y, random_state=42)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(clf.score(y_test, y_pred))
    return scores


# test using 3 fold crossvalidation
def test_crossval(clf):
    scores = []
    for idx_data in range(3):
        X = train_dna[idx_data]
        y = label_dna[idx_data]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            cv_scores.append(clf.score(y_test, y_pred))
        scores.append(np.mean(cv_scores))
    return scores


if __name__ == "__main__":
    # load data
    train_files = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
    test_files = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
    label_files = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']

    train_dna = np.asarray([_get_data(filename) for filename in train_files])
    test_dna = np.asarray([_get_data(filename) for filename in test_files])

    label_dna = np.asarray([np.asarray(_get_data(filename, header=True))[
                           :, 1] for filename in label_files])

    # best parameters
    lbda = 0.001
    k = 5
    m = 1
    clf1 = KLR(lbda, kernel='mismatch_kernel', k=k, m=m)
    print ("training first classifier...")
    clf1.fit(train_dna[0], label_dna[0])

    lbda = 0.0001
    k = 6
    m = 1
    print ("training second classifier...")
    clf2 = KLR(lbda, kernel='mismatch_kernel', k=k, m=m)
    clf2.fit(train_dna[1], label_dna[1])

    lbda = .0001
    print ("training third classifier...")
    clf3 = SVM(lbda, kernel='spectrum_norm_kernel')
    clf3.fit(train_dna[2], label_dna[2])

    # predict
    print ("predicting test data...")
    ypreds = [clf1.predict(test_dna[0]), clf2.predict(test_dna[1]), clf3.predict(test_dna[2])]

    allpreds = pd.DataFrame(np.atleast_1d(ypreds).flatten(), columns=['Bound'])
    allpreds.index.name = 'Id'
    allpreds = allpreds.reset_index()

    # save to csv file
    print ("saving predicted values...")
    allpreds.to_csv('submission_kaggle.csv', index=False)
