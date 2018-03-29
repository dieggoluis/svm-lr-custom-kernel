import cvxopt
import numpy as np
from sklearn.base import BaseEstimator
from kernels import mismatch_kernel, spectrum_kernel, spectrum_norm_kernel
from kernels import substringKernel_cpp_nystrom, substringKernel_cpp


def qp(P, q, A, b, C, verbose=True):
    n = P.shape[0]
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(np.concatenate(
        [np.diag(-np.ones(n)), np.diag(np.ones(n))], axis=0))
    h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)], axis=0))

    # Solve QP problem
    cvxopt.solvers.options['show_progress'] = verbose
    if A is None or b is None:
        solution = cvxopt.solvers.qp(P, q, G, h, solver='mosec')
    else:
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b, solver='mosec')

    return np.ravel(solution['x'])


class SVM (BaseEstimator):
    def __init__(self, lbda=1.0, kernel='spectrum_kernel', **kargs):
        self.lbda = lbda
        self.kargs = kargs
        self.kernel_ = kernel
        if kernel == 'mismatch_kernel':
            self.kernel = mismatch_kernel
        elif kernel == 'spectrum_kernel':
            self.kernel = spectrum_kernel
        elif kernel == 'spectrum_norm_kernel':
            self.kernel = spectrum_norm_kernel
        elif kernel == 'substring_kernel':
            self.kernel = substringKernel_cpp_nystrom
            self.test_kernel = substringKernel_cpp
        else:
            print ('Invalid kernel')


    def _dual_solver(self, K, C, y):
        n = K.shape[0]
        P = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                P[i][j] = K[i][j] * y[i] * y[j]
        q = -np.ones(n)
        A = np.reshape(1. * y, (1, n))
        b = .0
        alpha = qp(P, q, A, b, C, verbose=False)
        return alpha

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        self.y_train[self.y_train == 0] = -1

        n = X_train.shape[0]
        C = 1 / (2 * self.lbda * n)
        K = self.kernel(X_train, X_train, **self.kargs)

        # solve optimization problem
        self._alpha = self._dual_solver(K, C, self.y_train)

        # index for support vector
        idx = np.argmax(
            np.min([np.abs(self._alpha), np.abs(self._alpha - C)], axis=0))
        self.bias = self._alpha.dot(K[:, idx] * self.y_train) - self.y_train[idx]
        return self

    def predict(self, X_test):
        if self.kernel_ != 'substring_kernel':
            K = self.kernel(self.X_train, X_test, **self.kargs)
        else:
            K = self.test_kernel(self.X_train, X_test, **self.kargs)
        y_pred = np.sign(np.dot(K.T, self._alpha * self.y_train) - self.bias).astype(int)
        y_pred[y_pred < 0] = 0
        return y_pred

    def score(self, y_true, y_pred):
        return np.sum(y_pred == y_true).astype(float) / np.size(y_true)
