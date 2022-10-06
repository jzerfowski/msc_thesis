from itertools import compress
from multiprocessing import Process
from typing import List, Union, Optional
import numpy as np
import math

from DataProcessor import DataProcessor, check_data_dimensions, T_Data, \
    T_Timestamps

import logging

logger = logging.getLogger(__name__)


class ProcessingPipelineException(Exception):
    pass


class ProcessingPipeline(DataProcessor):
    def __init__(self, in_channel_labels: Union[List[str], int], nodes: Union[List[dict], List[DataProcessor]], in_feature_dims: List[int] = None, **settings):
        if isinstance(in_channel_labels, int):
            in_channel_labels = [f"In-{i:02}" for i in range(in_channel_labels)]

        super().__init__(in_channel_labels=in_channel_labels, in_feature_dims=in_feature_dims, **settings)

        if not isinstance(nodes, list):
            raise NotImplementedError("ProcessingPipeline's nodes argument needs argument 'nodes' to be a list")

        if all([isinstance(node, dict) for node in nodes]):
            self.nodes: List[DataProcessor] = []
            intermediate_channel_labels = in_channel_labels
            intermediate_feature_dims = self.in_feature_dims

            for node_settings in nodes:
                try:
                    module = getattr(DataProcessors, node_settings["type"])
                    if 'in_channel_labels' in node_settings:
                        intermediate_channel_labels = node_settings['in_channel_labels']
                        del node_settings['in_channel_labels']
                    if 'in_feature_dims' in node_settings:
                        assert(intermediate_feature_dims == node_settings['in_feature_dims'], "in_feature_dims must be consistent")
                        del node_settings['in_feature_dims']
                    node: DataProcessor = getattr(module, node_settings["type"])(in_channel_labels=intermediate_channel_labels, in_feature_dims=intermediate_feature_dims, **node_settings)
                    self.nodes.append(node)
                except TypeError as e:
                    raise ProcessingPipelineException(f"An error occured when initializing a node of type {node_settings['type']} with in_feature_dims={intermediate_feature_dims} and settings {node_settings}:\n{e}")

                intermediate_channel_labels = node.out_channel_labels
                intermediate_feature_dims = node.out_feature_dims
            out_channel_labels = intermediate_channel_labels
            self._nodes_init: List[dict] = nodes
            logger.info(f"Loaded pipeline with {len(self.nodes)} nodes from settings in dictionaries.")


        elif all([isinstance(node, DataProcessor) for node in nodes]):
            self.nodes: List[DataProcessor] = nodes
            if not in_channel_labels == self.nodes[0].in_channel_labels:
                raise ValueError(
                    f"ProcessingPipeline's in_channel_labels are different from first node's in_channel_labels:\n"
                    f"({self.in_channel_labels} vs. {self.nodes[0].in_channel_labels}")
            out_channel_labels = self.nodes[-1].out_channel_labels
            self._nodes_init: List[dict] = [node.settings for node in nodes]
            logger.info(f"Loaded pipeline with {len(self.nodes)} nodes from list of DataProcessors.")

        else:
            raise NotImplementedError(
                "ProcessingPipeline requires 'nodes' to be a list of only dictionaries or only DataProcessors")

        self.out_channel_labels = out_channel_labels
        self.out_feature_dims = self.nodes[-1].out_feature_dims

    @check_data_dimensions
    def process(self, data: T_Data, timestamps: T_Timestamps = None, *args: any, **kwargs: any) -> (
            T_Data, T_Timestamps):
        """Propagate the data through all classifier nodes
        :param data: shape (n_trials, n_channels, ..., n_times)
        :param timestamps:
        :return: data of shape (n_trials, ...) where the last dimensions depend on the nodes
        """
        for node in self.nodes:
            data, timestamps = node.process(data, timestamps, *args, **kwargs)
        return data, timestamps

    def process_trials(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None, clear_between: bool = True, chunk_size: int = 50, *args, **kwargs):
        """
        Propagates a number of trials through the pipeline by splitting it into multiple long chunks
        Timestamps are only processed if they have the same length the the n_times dimension of the data
        :param data: shape (n_trials, n_channels, ..., n_times)
        :param timestamps: list of timestamps
        :param clear_between: indicates if pipeline.clear() should be called between trials. Can result in return values of unequal length if False
        :param chunk_size: chunk size in n_times dimension for using with process()
        :param args:
        :param kwargs:
        :return: data_out, timestamps_out retrieved from process_chunks
        """
        n_trials, n_channels, *n_features, n_times = data.shape

        data_out = []
        for trial_i in range(n_trials):
            if clear_between:
                self.clear()
            data_trial_in = data[trial_i:trial_i+1, :]  # slicing to preserve dimensions
            data_trial_out, timestamps_out = self.process_chunks(data_trial_in, timestamps, chunk_size=chunk_size, *args, **kwargs)
            data_out.append(data_trial_out)

        if clear_between:
            self.clear()

        data_out = np.concatenate(data_out, axis=0)

        return data_out, timestamps_out

    def process_chunks(self, data: np.ndarray, timestamps=None, chunk_size: int = 10, clean_none: bool = True, *args, **kwargs) -> (list, list):
        """Propagates large/unknown amounts of data through the pipeline by splitting it into chunks and collecting the
        resulting data and timestamps in two lists.
        Timestamps are only processed if they have the same length as the n_times dimension of data.
        :param data: shape (n_trials, n_channels, ..., n_times)
        :param timestamps: list of timestamps
        :param chunk_size: int
        :param clean_none: delete chunks from the return which are None
        :param args:
        :param kwargs:
        :return: (List of processed data, List of processed timestamps)
        """
        n_trials, n_channels, *n_features, n_times = data.shape

        n_chunks: int = math.ceil(n_times / chunk_size)

        processed_data = []
        processed_timestamps = []

        include_timestamps = (timestamps is not None) and \
                             (hasattr(timestamps, '__iter__')) and \
                             (n_times == len(timestamps))

        progress_updates = 5
        for i_chunk in range(n_chunks):
            if progress_updates and i_chunk % (n_chunks//progress_updates) == 0:
                logger.debug(f"Processing chunk {i_chunk}/{n_chunks} ({100*i_chunk//n_chunks}%)")

            start_sample: int = i_chunk * chunk_size
            end_sample: int = min(n_times, start_sample + chunk_size)

            chunk: np.array = data[..., start_sample:end_sample]
            chunk_timestamps = timestamps[start_sample:end_sample] if include_timestamps else None

            processed_chunk, processed_chunk_timestamps = self.process(chunk, chunk_timestamps, *args, **kwargs)

            processed_data.append(processed_chunk)
            processed_timestamps.append(processed_chunk_timestamps)

        if clean_none:
            mask = [x is not None for x in processed_data]
            # Use compress to apply the mask on the processed data
            processed_data = np.concatenate(list(compress(processed_data, mask)), axis=-1)
            processed_timestamps = np.array(list(compress(processed_timestamps, mask))).squeeze().T.squeeze()

        return processed_data, processed_timestamps

    def train(self, data, labels, timestamps=None, *args, **kwargs):
        data_processed, labels_processed, timestamps_processed = data, labels, timestamps
        for node in self.nodes:
            logger.debug(f"Training Node {node}")
            (data_processed, labels_processed, timestamps_processed) = node.train(data_processed, labels_processed, timestamps_processed, *args, **kwargs)

        # Important call which will reset the states of all nodes after training
        self.clear()

        return data_processed, labels_processed, timestamps_processed

    def get_class_for_value(self, values):
        for node in self.nodes[::-1]:
            if hasattr(node, 'get_class_for_value'):
                return node.get_class_for_value(values)
        raise AttributeError("ProcessingPipeline does not contain a node with method get_class_for_value()")

    def clear(self, *args, **kwargs):
        for node in self.nodes:
            node.clear(*args, *kwargs)
        logger.debug("All nodes in ProcessingPipeline cleared")

    def close(self, *args, **kwargs):
        for node in self.nodes:
            node.close(*args, **kwargs)
        logger.debug("All nodes in ProcessingPipeline closed")

    def __str__(self):
        classifiers = "\n".join(
            [f"{node} ({node.num_in_channels}Ch --> {node.num_out_channels}Ch)" for node in self.nodes])
        return (f"ProcessingPipeline with {len(self.nodes)} nodes:\n"
                f"{classifiers}")

    def __repr__(self):
        return self.__str__()

    def get_settings(self, *args, **kwargs):
        settings = super().get_settings(*args, **kwargs)
        nodes_dict = [node.get_settings(*args, **kwargs) for node in self.nodes]
        settings.update(nodes=nodes_dict, _nodes_init=self._nodes_init)
        return settings