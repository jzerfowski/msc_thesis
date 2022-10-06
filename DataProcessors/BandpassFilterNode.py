from DataProcessor import check_data_dimensions, T_Timestamps, T_Data, \
    clear_decorator
from ProcessingNode import ProcessingNode

from typing import Union, List

import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class BandpassFilterNode(ProcessingNode):
    def __init__(self, in_channel_labels: List[str], sfreq: float, filter_length: int = 499, f_highpass: float = 5,
                 f_lowpass: float = 30, **settings):
        super().__init__(in_channel_labels, **settings)

        self.filter_length: int = None
        self.sfreq: float = None
        self.f_highpass: float = None
        self.f_lowpass: float = None

        self.filter_b = None
        self.filter_a = None
        self.zf = None

        self._init_filter(sfreq=sfreq, filter_length=filter_length, f_highpass=f_highpass, f_lowpass=f_lowpass)

    def _init_filter(self, sfreq: float, filter_length: int, f_highpass: float,
                     f_lowpass: float):

        self.zf = None
        self.sfreq = sfreq
        self.filter_length = filter_length
        self.f_highpass = f_highpass
        self.f_lowpass = f_lowpass

        self.filter_b = scipy.signal.firwin(self.filter_length, [self.f_highpass, self.f_lowpass], pass_zero=False,
                                            fs=self.sfreq)
        self.filter_a = 1

    @check_data_dimensions
    def process(self, data: T_Data, timestamps: T_Timestamps = None, *args: any, **kwargs: any) -> (
            T_Data, T_Timestamps):
        if data is None or data.shape[-1] == 0:
            return None, None

        # initialize filter with zeros if not initialized yet
        if self.zf is None:
            self.zf = np.zeros([1] + list(data.shape)[1:-1] + [self.filter_length - 1])

        # filter data
        data, self.zf = scipy.signal.lfilter(self.filter_b, self.filter_a, data, axis=-1, zi=self.zf)

        return data, timestamps

    @clear_decorator(before=True, after=True)
    def train(self, data: T_Data, labels: np.ndarray, timestamps=None, *args: any, sfreq: float = None,
              filter_length: int = None, f_highpass: float = None, f_lowpass: float = None, **kwargs: any) -> (
            Union[T_Data, List[T_Data]]):

        logger.debug(f"Calling train() on BandpassFilterNode with {sfreq=}, {filter_length=}, {f_highpass=}, {f_lowpass=}")
        call_init_filter = False
        if filter_length is None:
            filter_length = self.filter_length
            call_init_filter = True

        if f_highpass is None:
            f_highpass = self.f_highpass
            call_init_filter = True

        if f_lowpass is None:
            f_lowpass = self.f_lowpass
            call_init_filter = True

        if sfreq is None:
            sfreq = self.sfreq
            call_init_filter = True

        if call_init_filter:
            self._init_filter(sfreq=sfreq, filter_length=filter_length, f_highpass=f_highpass, f_lowpass=f_lowpass)
            logger.debug(f"Retrained BandpassFilterNode with {sfreq=}, {filter_length=}, {f_highpass=}, {f_lowpass=}")

        self.clear()
        data_out, timestamps_out = self.process(data, timestamps)
        return data_out, labels, timestamps_out

    def clear(self, *args, **kwargs):
        self.zf = None

    def plot_filter_response(self, ax=None):
        if ax is None:
            ax = plt.gca()
        w, h = scipy.signal.freqz(b=self.filter_b, a=self.filter_a, fs=self.sfreq)

        ax.plot(w, 20 * np.log10(np.abs(h)))
        ax.set_title('Digital filter frequency response')
        ax.set_ylabel('Amplitude Response [dB]')

        ax.set_xlabel('Frequency [Hz]')
        ax.grid()
        return ax

    def get_settings(self, *args, **kwargs):
        settings = super().get_settings(*args, **kwargs)
        settings['sfreq'] = self.sfreq
        settings['filter_length'] = self.filter_length
        settings['f_highpass'] = self.f_highpass
        settings['f_lowpass'] = self.f_lowpass

        return settings


if __name__ == '__main__':
    num_in_channels = 5
    node = BandpassFilterNode(in_channel_labels=[f"Ch{i}" for i in range(num_in_channels)], sfreq=1000, f_highpass=10,
                              f_lowpass=14)
    node.plot_filter_response()
