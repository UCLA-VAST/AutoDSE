import heapq as pq
from .gradient import GradientAlgorithm, ExploreTreeNode, ParamWPointBatch, get_batch_name
from ..parameter import (DesignParameter, DesignPoint, DesignSpace, MerlinParameter,
                         gen_key_from_design_point, get_default_point)
from collections import OrderedDict
from typing import Dict, Generator, List, NamedTuple, Optional, Set, Tuple
from ..result import HierPathNode, HLSResult, MerlinResult, Result
from ..dsproc.dsproc import topo_sort_param_ids


class HybridAlgorithm(GradientAlgorithm):              
    def __init__(self,
                 ds: DesignSpace,
                 latency_thd: float = 64,
                 fg_first: bool = True,
                 quality_type: str = 'resource-efficiency',
                 comp_order: Optional[List[str]] = None,
                 comm_order: Optional[List[str]] = None,
                 log_file_name: str = 'algo.log',
                 hybrid_q_thd: float = 0,
                 hybrid_num_points: int = 50):
        """Constructor of gradient algorithm.

        Args:
            ds: Given design space.
            latency_thd: Latency threshold.
            fg_first: If the gradient should focus on fine-grained loops first.
            quality_type: The metric to judge result qualities.
            comp_order: The design parameter order for compute-bound bottleneck.
            comm_order: The design parameter order for memory-bound bottleneck.
            log_file_name: Name of the log file.
            hybrid_q_thd: the least improve in quality needed to run exhaustive exploration
            hybrid_num_points: numbers of points to run the exhaustive exploration
        """
        super(GradientAlgorithm, self).__init__(ds, log_file_name)
        self.latency_thd = latency_thd
        self.fg_first = fg_first
        self.quality_type = quality_type
        self.hybrid_q_thd = hybrid_q_thd
        self.hybrid_num_points = hybrid_num_points

        self.comp_order = comp_order if comp_order else ['PARALLEL', 'PIPELINE']
        self.comm_order = comm_order if comm_order else [
            'INTERFACE', 'CACHE', 'PIPELINE', 'TILE', 'TILING'
        ]
        
        self.log.info('Computation order (high to low): %s',
                      [p for p in self.comp_order])
        self.log.info('Communication order (high to low): %s',
                      [p for p in self.comm_order])
        self.ordered_pids = topo_sort_param_ids(ds)

        # Build scope map
        self.scope2param: Dict[str, List[DesignParameter]] = {}
        for param in ds.values():
            for scope in param.scope:
                if scope not in self.scope2param:
                    self.scope2param[scope] = []
                self.scope2param[scope].append(param)

    
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

        self.log.info('Launch hybrid search algorithm')
        
        prev_best_quality = {} ## {batch_name: quality}

        # Start from the default point
        curr_point = get_default_point(self.ds)
        curr_key = gen_key_from_design_point(curr_point)
        key_n_results = yield [curr_point]
        assert key_n_results is not None
        curr_result = list(key_n_results.values())[0]

        if not curr_result.valid:
            self.log.warning('Default point is invalid: %s', str(curr_point))
            self.log.warning('This exploration is low confidence to achieve high QoR.')

        focus_params = self.get_hotspot_params(curr_result, set())
        self.log.info('Focus parameters (high order goes to tail): %s',
                      [p.name for p in focus_params])
        child = self.gen_child_points(curr_point, focus_params)
        flatten_child = self.gen_flatten_points(curr_point)
        if flatten_child:
            child.append(flatten_child)

        # Push the default point as the tree node
        explore_tree: List[List[Tuple[float, ExploreTreeNode]]] = [[]]
        tuned: Set[str] = set()
        pq.heappush(explore_tree[0],
                    (0, ExploreTreeNode(curr_key, 0, curr_result.perf, curr_result, child, tuned)))

        iter_cnt = 0
        while True:
            iter_cnt += 1
            self.log.info('=== ITERATION %d ===', iter_cnt)

            # Identify the next available level with tunable parameters
            curr_lv = -1
            while -curr_lv <= len(explore_tree) and not explore_tree[curr_lv]:
                curr_lv -= 1
            if -curr_lv > len(explore_tree):
                self.log.info('Explore tree is totally empty, stop.')
                return
            self.log.info('Tree Level %d', len(explore_tree) + curr_lv)

            # Take the next point with the best cost to explore
            # Note: we only peek the first node in the heap without popping it
            _, curr_node = explore_tree[curr_lv][0]
            tuned = curr_node.tuned

            # Get the focus parameters to be tuned
            # Note: the child is a stack so we always pick the top (last) value.
            if not curr_node.child:
                break
            param_w_points = curr_node.child[-1]

            self.log.info('Working Node:')
            self.log_node(curr_node)

            if param_w_points.batches:
                working_batch_name = get_batch_name(param_w_points)
                self.log.info('Working batch: %s', working_batch_name)
                key_2_tags: Dict[str, str] = {}
                pending: List[DesignPoint] = []

                # Consume one batch and evaluate it
                curr_batch = param_w_points.batches.pop()
                for tag, point in curr_batch:
                    key_2_tags[gen_key_from_design_point(point)] = tag
                    pending.append(point)
                self.log.info('Evaluating %d poitns', len(pending))
                key_n_results = yield pending
                assert key_n_results is not None
                self.log.info('Received results from evaluator')

                # Create the next level if needed
                if curr_lv == -1:
                    explore_tree.append([])
                    curr_lv -= 1

                # Process the results and generate next points
                self.log.info('Results')
                for key, result in key_n_results.items():
                    quality = self.compute_quality(curr_result, result)
                    self.log_result(key_2_tags[key], quality, result)
                                         

                    if quality == -float('inf'):  # Evaluation failure or point-of-not-interest
                        self.log.info('-> No quality, abandon')
                        continue

                    assert result.valid
                    assert result.point is not None
                    
                    
                    # Add current parameter itself to tuned parameter
                    new_tuned = set(tuned)
                    if param_w_points.param:
                        new_tuned.add(param_w_points.param.name)
                        
                    # Initialize the best quality values
                    if working_batch_name not in prev_best_quality:
                        ## add key
                        prev_best_quality[working_batch_name] = quality
                    
                    prev_quality_value = prev_best_quality[working_batch_name]
                    if (quality - prev_quality_value) / prev_quality_value >= self.hybrid_q_thd:
                        prev_best_quality[working_batch_name] = quality
                        # run exhaustive for hybrid_num_points iteration
                        hybrid_iter = 0
                        while True and (hybrid_iter < self.hybrid_num_points):
                            self.log.info('Launch exhaustive search algorithm on point', result.point)

                            traverser = self.traverse(result.point, 0)
                            while True:
                                next_points: List[DesignPoint] = []
                                try:
                                    hybrid_iter += 1
                                    self.log.info('Exhaustive Iteration %d', hybrid_iter)
                                    next_points.append(next(traverser))
                                    self.log.info('%s', str(next_points[-1]))
                                    yield next_points
                                except StopIteration:
                                    if next_points:
                                        yield next_points
                                    break
                                
                                if hybrid_iter >= self.hybrid_num_points:
                                    self.log.info('Reached the limit for hybrid number of points, stop exhaustive.')
                                    break

                            self.log.info('No more points to be explored, stop exhaustive.')
                            
                    # Identify the hotspot
                    focus_params = self.get_hotspot_params(result, new_tuned)
                    if not focus_params:
                        self.log.info('-> No tunable parameters, abandon')
                        continue
                    # Gnerate next points and add to the tree
                    child = self.gen_child_points(result.point, focus_params)
                    pq.heappush(
                        explore_tree[curr_lv + 1],
                        (-quality,
                         ExploreTreeNode(key, quality, result.perf, result, child, new_tuned)))
                    self.log.info('-> Add to level %d with %d tunable parameters: %s',
                                  len(explore_tree) + curr_lv + 1, len(child),
                                  [p.name for p in focus_params])
            else:
                self.log.info('No batch to be worked')

            # Finished batches related this parameter
            if not param_w_points.batches:  # No more batches
                self.log.info('Run out of batches of parameter %s',
                              param_w_points.param.name if param_w_points.param else 'builtin')
                curr_node.child.pop()

            # Finished all child of this node
            if not curr_node.child:
                pq.heappop(explore_tree[curr_lv])
                self.log.info('Finished exploring current node, remove from the tree')

        self.log.info('No more points to be explored, stop.')