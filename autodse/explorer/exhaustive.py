"""
The exhaustive search algorithm
"""
from typing import Dict, Generator, List, Optional

from ..dsproc.dsproc import topo_sort_param_ids
from ..parameter import DesignPoint, DesignSpace, get_default_point
from ..result import Result
from .algorithm import SearchAlgorithm


class ExhaustiveAlgorithm(SearchAlgorithm):
    """Exhaustively explore the design space.

    The order is based on the topological order of design parameters. Considering the
    evaluation overhead, we let users configure the batch size for evaluation.

    Attributes:
        batch_size: The batch size of producing design points.
        ordered_pids: Design parameters in topological sort with cycle breaking.
    """

    def __init__(self, ds: DesignSpace, batch_size: int = 8, log_file_name: str = 'algo.log'):
        """Constructor.

        Args:
            ds: Design space.
            batch_size: The batch size of producing design points.
            log_file_name: The name of log file.
        """
        super(ExhaustiveAlgorithm, self).__init__(ds, log_file_name)
        self.batch_size = batch_size
        self.ordered_pids = topo_sort_param_ids(ds)

    def traverse(self, point: DesignPoint, idx: int) -> Generator[DesignPoint, None, None]:
        """DFS traverse the design space and yield leaf points.

        Args:
            point: The current design point.
            idx: The current manipulated parameter index.

        Returns:
            A resursive generator for traversing.
        """

        if idx == len(self.ordered_pids):
            # Finish a point
            yield point
        else:
            yield from self.traverse(point, idx + 1)

            # Manipulate idx-th point
            new_point = self.clone_point(point)
            while self.move_by(new_point, self.ordered_pids[idx]) == 1:
                yield from self.traverse(new_point, idx + 1)
                new_point = self.clone_point(new_point)

    def gen(self) -> Generator[List[DesignPoint], Optional[Dict[str, Result]], None]:
        #pylint:disable=missing-docstring

        self.log.info('Launch exhaustive search algorithm')

        traverser = self.traverse(get_default_point(self.ds), 0)
        iter_cnt = 0
        while True:
            next_points: List[DesignPoint] = []
            try:
                iter_cnt += 1
                self.log.info('Iteration %d', iter_cnt)
                while len(next_points) < self.batch_size:
                    next_points.append(next(traverser))
                    self.log.info('%s', str(next_points[-1]))
                yield next_points
            except StopIteration:
                if next_points:
                    yield next_points
                break

        self.log.info('No more points to be explored, stop.')
