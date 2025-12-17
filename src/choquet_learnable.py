import warnings

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize

from fuzzy_measure.classical import *
from fuzzy_measure.tnorm import *
from optimize.gradient_descent import GD_minimize
from optimize.objective_functions import objective_tnorm, objective
from utils import preprocess

class Choquet_APB(ClassifierMixin, BaseEstimator):
    """Choquet Classifier APB (Algebraic Product Binary) - Classical"""

    def __init__(
        self, methode="Power", optimizer="Nelder-Mead", process_data=True, jac=True
    ) -> None:
        super().__init__()
        self.methode = methode
        self.optimizer = optimizer
        self.y_pred = None
        self.y_est = None
        self.theta = None
        self.out_ = None
        self.success = False
        self.process_data = process_data
        self.jac = jac

    def initialize(self, X, y=None):
        n = X.shape[0]
        self.p_fit = X.shape[1]
        if self.methode == "Power":
            self.theta = 0.5
        elif self.methode == "Weight":
            self.theta = np.ones(self.p_fit)
        else:
            raise ValueError("Methode must be Power or Weight")
        if self.process_data:
            self.X = preprocess(X, n)
        self.y_train = y

    def fit(self, X, y=None):
        self.initialize(X, y)
        self.out_ = minimize(
            objective,
            self.theta,
            args=(self.X, self.y_train, self.p_fit, self.methode),
            method=self.optimizer,
            jac=self.jac,
        )
        if (
            (self.out_.nit > 0)
            & ~(self.out_.success)
            & np.logical_and(self.out_.x, self.theta).sum()
        ):
            self.success = True
            warnings.warn(f"Optimization failed: {self.out_.message}")
        else:
            self.success = self.out_.success

        self.theta = self.out_.x

    def predict_proba(self, X):
        n = X.shape[0]
        self.p_prd = X.shape[1]
        if self.success:
            if self.process_data:
                self.X_test = preprocess(X, n)
            if self.methode == "Power":
                self.y_est = fuzzy_power(self.X_test, self.p_prd, self.theta)
            elif self.methode == "Weight":
                self.y_est = fuzzy_weight(self.X_test, self.p_prd, self.theta)
        else:
            raise ValueError("Model not fitted")
        return np.vstack([1 - self.y_est, self.y_est]).T

    def predict(self, X):
        if self.success:
            if self.y_est is None:
                self.predict_proba(X)
            self.y_pred = np.where(self.y_est > 0.5, 1, 0)
        else:
            raise ValueError("Model not fitted")
        return self.y_pred

    def score(self, X, y=None):
        if self.success:
            if self.y_pred is None:
                self.predict(X)
            self.accuracy_ = accuracy_score(y, self.y_pred)
            self.f1_score_ = f1_score(y, self.y_pred)
            return self.accuracy_
        else:
            raise ValueError("Model not fitted")


class Choquet_APB_TNORM(ClassifierMixin, BaseEstimator):
    """Choquet Classifier APB (Algebraic Product Binary) - With T-Norms"""

    def __init__(
        self,
        methode="Power",
        optimizer="Nelder-Mead",
        process_data=True,
        jac=True,
        tnorm_c=3,
        alpha=0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.methode = methode
        self.optimizer = optimizer
        self.y_pred = None
        self.y_est = None
        self.theta = None
        self.out_ = None
        self.success = False
        self.process_data = process_data
        self.jac = jac
        self.tnorm_c = tnorm_c
        self.init_alpha = alpha
        self.alpha = None
        self.kwargs = kwargs

    def initialize(self, X, y=None):
        n = X.shape[0]
        self.p_fit = X.shape[1]
        if self.init_alpha is None:
            self.alpha = np.random.rand(1)[0]
        else:
            self.alpha = self.init_alpha
        if self.methode == "Power":
            self.theta = np.array([np.random.rand(1)[0], self.alpha])
            self.bnds = [(-7, 7), (-5, 5)]
        elif self.methode == "Weight":
            self.theta = np.concatenate([np.ones((self.p_fit)), [self.alpha]])
            self.bnds = [(0, None)] * (len(self.theta) - 1) + [(-5, 5)]
        else:
            raise ValueError("Methode must be Power or Weight")
        if self.process_data:
            self.X = preprocess(X, n)
        y = y.ravel()
        self.label_encoder = LabelEncoder().fit(y)
        y = self.label_encoder.transform(y)
        self.y_train = y
        self.classes_ = self.label_encoder.classes_
        self.encoded_labels = self.label_encoder.transform(self.label_encoder.classes_)

    def fit(self, X, y=None):
        self.initialize(X, y)
        if self.optimizer == "GD":
            self.out_ = GD_minimize(
                self.X,
                self.y_train,
                self.theta,
                self.p_fit,
                self.tnorm_c,
                self.methode,
                objective_func=objective_tnorm,
                **self.kwargs,
            )
        else:
            self.out_ = minimize(
                objective_tnorm,
                self.theta,
                args=(
                    self.X,
                    self.y_train,
                    self.p_fit,
                    self.tnorm_c,
                    self.methode,
                ),
                method=self.optimizer,
                jac=self.jac,
                bounds=self.bnds,
            )
        if (
            (self.out_["nit"] > 0)
            & ~(self.out_["success"])
            & np.logical_and(self.out_["x"], self.theta).sum()
        ):
            self.out_["success"] = True
            warnings.warn(f"Optimization failed: {self.out_['message']}")
        else:
            self.success = self.out_["success"]

        self.weight = self.out_["x"][:-1]
        self.beta = self.out_["x"][-1]

    def predict_proba(self, X):
        n = X.shape[0]
        self.p_prd = X.shape[1]
        if self.success:
            if self.process_data:
                self.X_test = preprocess(X, n)
            if self.methode == "Power":
                self.y_est = fuzzy_power_tnorm(
                    self.X_test, self.p_prd, self.weight, self.beta, self.tnorm_c
                )
            elif self.methode == "Weight":
                self.y_est = fuzzy_weight_tnorm(
                    self.X_test, self.p_prd, self.weight, self.beta, self.tnorm_c
                )
        else:
            raise ValueError("Model not fitted")
        return np.vstack([1 - self.y_est, self.y_est]).T

    def predict(self, X):
        if self.success:
            self.predict_proba(X)
            self.y_pred = np.where(self.y_est > 0.5, 1, 0)
        else:
            raise ValueError("Model not fitted")
        return self.y_pred

    def score(self, X, y=None):
        y = self.label_encoder.transform(y.ravel())
        if self.success:
            self.predict(X)
            self.accuracy_ = accuracy_score(y, self.y_pred)
            self.f1_score_ = f1_score(y, self.y_pred)
            return self.accuracy_
        else:
            raise ValueError("Model not fitted")

