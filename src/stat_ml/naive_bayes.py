import numpy as np


class NB:
    def __init__(self, n_categories, n_classes):
        self.category_cnt = [np.zeros((n_classes, n_cats)) for n_cats in n_categories]
        self.class_cnt = np.zeros(n_classes)

        self.n_categories = n_categories
        self.n_classes = n_classes

    def fit(self, X, y):
        self.feature_log_prob = []

        for i, cat_cnt in enumerate(self.category_cnt):
            for j, cnt in enumerate(cat_cnt):
                cnt += np.bincount(X[y == j, i], minlength=self.n_categories[i])

            # Laplacian smoothing

            smoothed = cat_cnt + 1
            self.feature_log_prob.append(
                np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
            )

        self.class_cnt += np.bincount(y, minlength=self.n_classes)
        self.class_log_prior = np.log(self.class_cnt) - np.log(self.class_cnt.sum())

        return self

    def predict(self, X):
        return np.argmax(self.joint_log_likelihood(X), axis=1)

    def joint_log_likelihood(self, X):
        jll = 0
        for ll, x in zip(self.feature_log_prob, X.T):
            jll += ll.T[x]

        return jll + self.class_log_prior
