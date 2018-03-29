from sklearn.base import BaseEstimator
import numpy as np
from kernels import mismatch_kernel, spectrum_kernel, spectrum_norm_kernel
from kernels import substringKernel_cpp_nystrom, substringKernel_cpp
from scipy.optimize import fmin_l_bfgs_b as lbfgs


class KLR (BaseEstimator):
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

    def _rgloss(self, x):
        return np.log(1. + np.exp(-x))

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def _rgloss_prime(self, x):
        return -self._sigmoid(-x)

    def _f(self, x):
        n = self._K.shape[0]
        reg = 0.5 * self.lbda * np.dot(np.dot(self._K, x), x)
        fx = np.sum(self._rgloss(self._y * np.dot(self._K, x)) / n) + reg
        return fx

    def _fprime(self, x):
        n = self._K.shape[0]
        yKx = self._y * np.dot(self._K, x)
        grad = np.dot(self._K.T, (self._y * self._rgloss_prime(yKx))
                      ) / n + self.lbda * np.dot(self._K, x)
        return grad

    def _solver(self, x0):
        alpha, _, _ = lbfgs(self._f, x0, fprime=self._fprime)
        return alpha

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.copy(y_train)
        self.y_train[self.y_train == 0] = -1
        self._y = self.y_train

        n = X_train.shape[0]
        self._K = self.kernel(X_train, X_train, **self.kargs)

        # solve optimization problem
        x0 = np.zeros(n)
        self._alpha = self._solver(x0)

        return self

    def predict(self, X_test):
        if self.kernel_ != 'substring_kernel':
            K = self.kernel(self.X_train, X_test, **self.kargs)
        else:
            K = self.test_kernel(self.X_train, X_test, **self.kargs)
        y_pred = np.sign(np.dot(K.T, self._alpha)).astype(int)
        y_pred[y_pred < 0] = 0
        return y_pred

    def score(self, y_true, y_pred):
        return np.sum(y_pred == y_true).astype(float) / len(y_true)
