"""
The module for making algorithm instance.
"""

from typing import Any, Dict

from .algorithm import SearchAlgorithm
from .exhaustive import ExhaustiveAlgorithm
from .gradient import GradientAlgorithm
from .hybrid import HybridAlgorithm
from ..logger import get_default_logger
from ..parameter import DesignSpace


class AlgorithmFactory():
    """Static class for registering and making algorithm instances"""

    @staticmethod
    def make(config: Dict[str, Any], ds: DesignSpace,
             log_file_name: str = 'algo.log') -> SearchAlgorithm:
        """Initialize a search algorithm based on the given configuration.

        Args:
            config: Configuration.
            ds: Design space.
            log_file_name: Name of the log file.

        Returns
            An object of initialized search algorithm.
        """

        log = get_default_logger('AlgorithmFactory')

        name = config['name']
        assert isinstance(name, str)
        if name == 'exhaustive':
            algo_config = config[name]
            assert isinstance(algo_config, dict)
            return ExhaustiveAlgorithm(ds=ds,
                                       log_file_name=log_file_name,
                                       batch_size=algo_config['batch-size'])
        if name == 'gradient':
            algo_config = config[name]
            assert isinstance(algo_config, dict)
            return GradientAlgorithm(ds=ds,
                                     latency_thd=algo_config['latency-threshold'],
                                     fg_first=algo_config['fine-grained-first'],
                                     quality_type=algo_config['quality-type'],
                                     comp_order=algo_config['compute-bound-order'],
                                     comm_order=algo_config['memory-bound-order'],
                                     log_file_name=log_file_name)
            
        if name == 'hybrid':
            algo_config = config['gradient']
            assert isinstance(algo_config, dict)
            return HybridAlgorithm(ds=ds,
                                     latency_thd=algo_config['latency-threshold'],
                                     fg_first=algo_config['fine-grained-first'],
                                     quality_type=algo_config['quality-type'],
                                     comp_order=algo_config['compute-bound-order'],
                                     comm_order=algo_config['memory-bound-order'],
                                     log_file_name=log_file_name)
        log.error('Unrecognized algorithm: %s', name)
        raise RuntimeError()
