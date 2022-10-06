from typing import List, Union

import numpy as np
import scipy
import logging

import sklearn.utils.multiclass

from DataProcessor import check_data_dimensions, T_Timestamps, T_Data
from ProcessingNode import ProcessingNode

logger = logging.getLogger(__name__)

Type_d = Union[List, np.ndarray]
Type_A = Union[List, np.ndarray]
Type_W = Union[List, np.ndarray]


class CSPNode(ProcessingNode):
    """
    This is not yet compatible with feature dimensions!
    """

    # def __init__(self, settings: dict, in_channel_labels: list):

    def __init__(self, in_channel_labels: List[str], n_components: int, method: str = 'outer', d: Type_d = None,
                 W: Type_W = None, A: Type_A = None, **settings):
        """
        method:
        :param in_channel_labels:
        :param n_components:
        :param method: Choose which eigenvectors should be selected; one of ['outer', 'largest', 'physiological']
        :param d:
        :param W:
        :param settings:
        """
        super().__init__(in_channel_labels, **settings)
        if n_components > self.num_in_channels:
            raise ValueError(f"n_components ({n_components}) must be <= num_in_channels ({self.num_in_channels})")

        if self.in_feature_dims:
            raise ValueError("CSPNode can not handle feature dimensions as of now")

        self.n_components = n_components

        self.method = method
        self.out_channel_labels = self.generate_out_channel_labels(num_out_channels=self.n_components)

        # Will call the appropriate setters which handle None
        self.d = d
        self.W = W
        self.A = A

        self.classes_ = None


    @check_data_dimensions
    def process(self, data: T_Data, timestamps: T_Timestamps=None, *args: any, **kwargs: any) -> (T_Data, T_Timestamps):
        if data is None:
            return None, None

        logger.debug(f"Processing data with shape {data.shape}, W.T has shape {self.W.T.shape}")
        return self.apply_csp(data, self.W), timestamps

    @classmethod
    def apply_csp(cls, data, W):
        return W.T @ data

    def train(self, data, labels, timestamps=None, *args, classes=None, compute_patterns=True, **kwargs):
        # data: shape (n_trials, n_channels, ..., n_times)

        d, W, A = self.csp_reduced(data, labels, classes, compute_patterns)

        self.d = d
        self.W = W
        self.A = A

        data_out, timestamps_out = self.process(data, timestamps)
        return data_out, labels, timestamps_out

        # return self.process(data, timestamps), self.get_settings(include_weights=True)

    def csp_reduced(self, data, labels, classes=None, compute_patterns=True):
        d, W = self.csp(data, labels, classes, normalize_weights=True)

        idxs = self.generate_component_indices(d, self.n_components)

        logger.debug(f"CSPNode training yielded {len(idxs)} eigenvalues: {d[idxs]}, {idxs=}")

        A = None
        if compute_patterns:
            A = self.compute_patterns(data, W, d)[:, idxs]

        return d[idxs], W[:, idxs], A

    def compute_patterns(self, data, W_full, d_full):
        cov_X = self.csp_cov(data)
        S = self.apply_csp(data, W_full)
        cov_S = self.csp_cov(S)
        A = cov_X @ W_full @ scipy.linalg.pinv(cov_S)
        return A

    def csp(self, data, labels, classes=None, normalize_weights=True):
        if classes is None:
            self.classes_ = sklearn.utils.multiclass.unique_labels(labels)
        else:
            self.classes_ = classes

        labels = np.array(labels)

        if len(self.classes_) <= 1:
            raise NotImplementedError("Cannot compute CSP on only one class!")

        cls_target, cls_nontarget = self.classes_[:2]

        cov_target = self.csp_cov(data[labels == cls_target])
        cov_nontarget = self.csp_cov(data[labels == cls_nontarget])

        d, W = scipy.linalg.eigh(a=cov_target, b=cov_target + cov_nontarget)

        if normalize_weights:
            W = W / np.linalg.norm(W, axis=0, keepdims=True)

        return d, W

    def csp_cov(self, data):
        n_trials, n_times = (data.shape[0], data.shape[-1])

        data -= np.mean(data, axis=-1, keepdims=True)  # Subtract the mean along time for each trial
        data = np.moveaxis(data, 0, -2)  # Move n_trials axis before n_times axis for later reshaping
        data = np.reshape(data, [-1, n_trials * n_times])  # reshape such that channel and feature axes are concatenated
        cov = np.cov(data, bias=True)  # Compute covariance matrix of shape (n_channels * (all n_features axes))^2

        return cov

    def generate_component_indices(self, d, n_components):
        if self.method == 'outer':
            n_non_target_comps = n_components // 2
            n_target_comps = n_components - n_non_target_comps
            # Compute the indices of the components which should be selected
            # the conversion to int is needed because for n_components=1 range(-1, -1, -1) would
            idxs = np.r_[np.arange(0, n_target_comps), np.arange(-1, -n_non_target_comps - 1, -1)]
            idxs = idxs[np.argsort(np.maximum(d[idxs], 1 - d[idxs]))][::-1]
        elif self.method == 'largest':
            idxs = np.argsort(np.maximum(d, 1 - d))[:-n_components - 1:-1]
        elif self.method == 'physiological':
            idxs = np.argsort(d)[:n_components]
        else:
            raise NotImplementedError(f"Method {self.method} is not implemented")
        return idxs

    @property
    def d(self) -> np.ndarray:
        return self._d

    @d.setter
    def d(self, d_arr: Type_d):
        if d_arr is None:
            logger.debug("Automatically setting default for d")
            d_arr = np.ones(self.n_components) / self.n_components
        elif not len(d_arr) == self.n_components:
            logger.warning(
                f"Length of d (={len(d_arr)}) does not match n_components (={self.n_components}. Setting automatically")
            d_arr = np.ones(self.n_components) / self.n_components
        else:
            d_arr = np.array(d_arr)  # make sure d contains a numpy array

        self._d = d_arr

    @property
    def W(self) -> np.ndarray:
        return self._W

    @W.setter
    def W(self, W_arr: Type_W):
        if W_arr is None:
            logger.debug("Automatically setting default for W")
            W_arr = np.ones((self.num_in_channels, self.n_components))
            W_arr /= np.linalg.norm(W_arr, axis=1, keepdims=True)
        elif not np.array(W_arr).shape == (self.num_in_channels, self.n_components):
            logger.warning(
                f"Shape of W (={np.array(W_arr).shape}) does not match needed shape "
                f"(={(self.num_in_channels, self.n_components)}. Setting automatic default")
            W_arr = np.ones((self.num_in_channels, self.n_components))
            W_arr /= np.linalg.norm(W_arr, axis=1, keepdims=True)
        else:
            W_arr = np.array(W_arr)

        self._W = W_arr

    @property
    def A(self) -> np.ndarray:
        return self._A

    @A.setter
    def A(self, A_arr: Type_A):
        if A_arr is None:
            logger.debug("Setting A to None")
            A_arr = None
        elif not np.array(A_arr).shape == (self.num_in_channels, self.n_components):
            logger.warning(
                f"Shape of A (={np.array(A_arr).shape}) does not match needed shape "
                f"(={(self.num_in_channels, self.n_components)}). Setting A to None")
            A_arr = None
        else:
            A_arr = np.array(A_arr)

        self._A = A_arr

    def get_settings(self, include_weights=True, include_patterns=True, *args, **kwargs):
        settings = super().get_settings(*args, **kwargs)
        if include_weights:
            settings.update(W=self.W.tolist(), d=self.d.tolist())
        if include_patterns:
            A = self.A.tolist() if self.A is not None else None
            settings.update(A=A)
        settings.update(method=self.method, n_components=self.n_components)
        return settings
