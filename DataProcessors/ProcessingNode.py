from DataProcessor import DataProcessor

from typing import List


class ProcessingNode(DataProcessor):
    def __init__(self, in_channel_labels: List[str], **settings):
        super().__init__(in_channel_labels, **settings)


    def get_settings(self, *args, **kwargs):
        return super().get_settings(*args, **kwargs)
