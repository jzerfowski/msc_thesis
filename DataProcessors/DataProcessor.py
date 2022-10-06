import functools
from typing import List, Union, Tuple, Iterable, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

T_Data = Union[np.ndarray, None]
T_Timestamps = Union[Iterable[float], float, None]
T_Settings = Union[dict, List[dict], None]


def check_data_dimensions(func):
    @functools.wraps(func)
    def wrapper_check_timestamp_dimensions(self, data: np.ndarray, timestamps: T_Timestamps = None, *args, **kwargs):
        if data is None:
            pass
        elif timestamps is None:
            pass
        elif isinstance(timestamps, (int, float)):
            # If there's only a single timestamp that is okay
            pass
        elif len(timestamps) == 1:
            # If there's only a single timestamps in a list that is okay
            pass
        elif not data.shape[-1] == len(timestamps):
            logger.warning(
                f"The times-dimension of data ({data.shape[-1]}) does not match the length of timestamps ({len(timestamps)})")
        return func(self, data=data, timestamps=timestamps, *args, **kwargs)

    return wrapper_check_timestamp_dimensions

def clear_decorator(before=True, after=True):
    """
    Call self.clear() before and/or after executing the function decorated by this
    """
    def wrapper_clear(func):
        @functools.wraps(func)
        def wrap_wrapper_clear(self, *args, **kwargs):
            if before: self.clear()
            returned = func(self, *args, **kwargs)
            if after: self.clear()
            return returned
        return wrap_wrapper_clear
    return wrapper_clear


class DataProcessor(object):
    # Nodes which provide .widget_dict for LiveWidget should set this to True
    has_widget = False

    def __init__(self, in_channel_labels: List[str], out_channel_labels: List[str] = None, in_feature_dims: Optional[List[int]] = None, **settings):
        self.in_channel_labels = in_channel_labels

        self._settings = settings

        self.out_channel_labels: List[str] = out_channel_labels
        if self.out_channel_labels is None:
            self.out_channel_labels = self.generate_out_channel_labels()

        self.in_feature_dims: List[int] = in_feature_dims
        if self.in_feature_dims is None:
            self.in_feature_dims = []

        self.out_feature_dims: List[int] = self.in_feature_dims

    @check_data_dimensions
    def process(self, data: T_Data, timestamps: T_Timestamps = None, *args: any, **kwargs: any) -> (
            T_Data, T_Timestamps):
        """
        Expects data in format (n_trials, n_channels,... n_times) (use np.moveaxis to achieve this shape)
        :param timestamps:
        :param data:
        :param *args:
        :param **kwargs:
        :return: should return data, timestamps
        """
        return data, timestamps

    @clear_decorator(before=True, after=True)
    def train(self, data: T_Data, labels: np.ndarray, timestamps=None, *args: any, **kwargs: any) \
            -> (Union[T_Data, List[T_Data]]):

        data_out, timestamps_out = self.process(data, timestamps)
        return data_out, labels, timestamps_out

    def generate_out_channel_labels(self, prefix=None, num_out_channels=None, **kwargs: any) -> List[str]:
        if prefix is None:
            prefix = self.__class__.__name__
        if num_out_channels is None:
            num_out_channels = self.num_in_channels
        digits = np.ceil(np.log10(num_out_channels)).astype(int)
        out_channel_labels = [f"{prefix}-{i:0{digits}}" for i in range(num_out_channels)]
        return out_channel_labels

    @property
    def num_in_channels(self):
        return len(self.in_channel_labels)

    @property
    def channel_count(self):
        return self.num_in_channels

    @property
    def num_out_channels(self):
        return len(self.out_channel_labels)

    @property
    def settings(self) -> T_Settings:
        return self.get_settings()

    def __str__(self):
        return f"{self.__class__.__name__}"

    def get_settings(self, in_channel_labels: bool = True, out_channel_labels: bool = True, in_feature_dims: bool = True, out_feature_dims: bool = True, *args,
                     **kwargs) -> T_Settings:
        """
        Return the settings relevant to the DataProcessor. This should be called by inheriting classes for construction
        of a complete settings dictionary
        The settings which no class in the hierarchy recognized are also appended, args and kwargs are ignored but
        offer flexibility in the future
        """
        settings = dict()

        # When called from a config we will have type as an argument in settings. When just the constructor of a node
        # is called we want to add information about its class to the settings
        if 'type' not in self._settings:
            settings.update(type=self.__class__.__name__)

        if in_channel_labels:
            settings.update(in_channel_labels=self.in_channel_labels)
        if out_channel_labels:
            settings.update(out_channel_labels=self.out_channel_labels)

        if in_feature_dims:
            settings.update(in_feature_dims=self.in_feature_dims)
        if out_feature_dims:
            settings.update(out_feature_dims=self.out_feature_dims)

        # For aesthetic reasons we would like to have type, in_ and out_channel_labels always in the front
        settings = {**settings, **self._settings.copy()}

        return settings

    def clear(self, *args, **kwargs):
        """
        Function to clear variables filled during runtime to reset a node
        For example a buffer might accumulate samples in train which should not be used in subsequent
        process()-calls. Can be used best by using the @clear_decorator(before:bool, after:bool)
        Should not reset weights set by train(), only variables modified in process()
        """
        pass

    def close(self, *args, **kwargs):
        """
        Function to close streams/files that are opened during runtime. E.g. used by LSLNode
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__} with {self.num_in_channels}Ch-->{self.num_out_channels}Ch"

    def __repr__(self):
        return self.__str__()

    @property
    def widget_dict(self) -> dict:
        """
        This property should return a widget_dict (for LivePlotWidget) when self.has_widget is True
        :return:
        """
        return None
