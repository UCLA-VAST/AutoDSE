"""
Design Space Processor
"""
from collections import deque
from copy import deepcopy
from logging import Logger
from typing import Deque, Dict, List, Optional, Set, Union

from ..logger import get_default_logger
from ..parameter import (DesignParameter, DesignPoint, DesignSpace, MerlinParameter,
                         create_design_parameter, get_default_point)
from ..util import safe_eval


def get_dsproc_logger() -> Logger:
    """Attach the logger of this module"""
    return get_default_logger('DSProc')


def compile_design_space(user_ds_config: Dict[str, Dict[str, Union[str, int]]],
                         scope_map: Optional[Dict[str, List[str]]]) -> Optional[DesignSpace]:
    """Compile the design space from the config JSON file.

    Args:
        user_ds_config: The input design space configure loaded from a JSON file.
                        Note that the duplicated ID checking should be done when
                        loading the JSON file and here we assume no duplications.
        scope_map: The scope map that maps design parameter ID to its scope.

    Returns:
        The design space compiled from the kernel code; or None if failed.
    """
    log = get_dsproc_logger()
    params: Dict[str, DesignParameter] = {}
    for param_id, param_config in user_ds_config.items():
        param = create_design_parameter(param_id, param_config, MerlinParameter)
        if param:
            if not isinstance(param, MerlinParameter) or param.ds_type not in [
                    'PARALLEL', 'PIPELINE', 'TILING', 'TILE'
            ]:
                param.scope.append('GLOBAL')
            else:
                if scope_map and param_id in scope_map:
                    param.scope = scope_map[param_id]
            params[param_id] = param

    error = check_design_space(params)
    if error > 0:
        log.error('Design space has %d errors', error)
        return None

    analyze_child_in_design_space(params)

    log.info('Design space contains %s valid design points', count_design_points(params))
    log.info('Finished design space compilation')
    return params


def count_design_points(ds: DesignSpace) -> int:
    """Count the valid points in a given design space.

    Args:
        ds: Design space to be counted.

    Returns:
        Number of valid design points.
    """

    log = get_dsproc_logger()

    def helper(ds: DesignSpace, sorted_ids: List[str], idx: int, point: DesignPoint) -> int:
        """Count the deisgn points of parameters by traversing topological sorted parameters."""

        # Reach to the end
        if idx == len(sorted_ids):
            return 1

        pid = sorted_ids[idx]
        param = ds[pid]
        options = safe_eval(param.option_expr, point)

        counter = 0
        if param.child:
            # Sum all points under each option
            for option in options:
                point[pid] = option
                counter += helper(ds, sorted_ids, idx + 1, point)
        else:
            # Product the number of options with the rest points
            counter = len(options) * helper(ds, sorted_ids, idx + 1, point)
        log.debug('Node %s: %d', pid, counter)
        return counter

    point = get_default_point(ds)
    sorted_ids = topo_sort_param_ids(ds)
    return helper(ds, sorted_ids, 0, point)


def check_design_space(params: DesignSpace) -> int:
    """Check design space for missing dependency and duplication.

    Args:
        params: The overall design space.

    Returns:
        The number of errors found in the design space.
    """

    log = get_dsproc_logger()
    error = 0

    for pid, param in params.items():
        has_error = False

        # Check dependencies
        for dep in param.deps:
            if dep == pid:
                log.error('Parameter %s cannot depend on itself', pid)
                error += 1
                has_error = True
            if dep not in params.keys():
                log.error('Parameter %s depends on %s which is undefined or not allowed', pid, dep)
                error += 1
                has_error = True

        if has_error:
            continue

        # Check expression types
        # Assign default value to dependent parameters
        local = {}
        for dep in param.deps:
            local[dep] = params[dep].default

        # Try to get an option list
        options: Optional[List[Union[int, str]]] = None
        try:
            options = safe_eval(param.option_expr, local)
        except (NameError, ValueError, TypeError, ZeroDivisionError) as err:
            log.error('Failed to get the options of parameter %s: %s', pid, str(err))
            error += 1

        # Try to get the order of options
        if options is not None and param.order and isinstance(param, MerlinParameter):
            for option in options:
                if safe_eval(param.order['expr'], {param.order['var']: option}) is None:
                    log.error('Failed to evaluate the order of option %s in parameter %s', option,
                              pid)
                    error += 1
    return error


def analyze_child_in_design_space(params: DesignSpace) -> None:
    """Traverse design parameter dependency and build child list for each parameter in place.

    Args:
        params: The overall design space
    """

    # Setup child for each parameter
    for pid, param in params.items():
        for dep in param.deps:
            params[dep].child.append(pid)

    # Remove duplications
    for param in params.values():
        param.child = list(dict.fromkeys(param.child))


def topo_sort_param_ids(space: DesignSpace) -> List[str]:
    """Sort the parameter IDs topologically.

    Args:
        space: The design space to be sorted.

    Returns:
        The sorted IDs.
    """

    def helper(curr_id: str, visited: Set[str], stack: List[str]) -> None:
        """The helper function for topological sort."""
        visited.add(curr_id)
        for dep in space[curr_id].deps:
            if dep not in visited:
                helper(dep, visited, stack)
        stack.append(curr_id)

    visited: Set[str] = set()
    stack: List[str] = []
    for pid in space.keys():
        if pid not in visited:
            helper(pid, visited, stack)
    return stack


def partition(space: DesignSpace, limit: int) -> Optional[List[DesignSpace]]:
    """Partition the given design space to at most the limit parts.

    Args:
        space: The design space to be partitioned.
        limit: The maximum number of partitions.

    Returns:
        The list of design space partitions.
    """
    log = get_dsproc_logger()
    sorted_ids = topo_sort_param_ids(space)

    part_queue = deque([deepcopy(space)])
    ptr = 0
    while len(part_queue) < limit and ptr < len(space):
        next_queue: Deque[DesignSpace] = deque()
        while part_queue:
            # Partition based on the current parameter
            curr_space = part_queue.pop()
            param_id = sorted_ids[ptr]
            param = curr_space[param_id]

            # Assign default value to dependent parameters
            local = {}
            for dep in param.deps:
                local[dep] = curr_space[dep].default

            # Evaluate the available options
            parts: Optional[Dict[int, List[Union[str, int]]]] = None
            if param.order and isinstance(param, MerlinParameter) and param.ds_type == 'PIPELINE':
                options = safe_eval(param.option_expr, local)
                if options is None:
                    log.error('Failed to evaluate options for parameter %s', param.name)
                    return None
                for option in options:
                    part_idx = safe_eval(param.order['expr'], {param.order['var']: option})
                    if part_idx is None:
                        log.error('Failed to evaluate the order of option %s in parameter %s',
                                  option, param.name)
                        return None
                    if parts is None:
                        parts = {}
                    if part_idx not in parts:
                        parts[part_idx] = []
                    parts[part_idx].append(option)

            accum_part = len(part_queue) + len(next_queue)
            if parts and len(parts) == 1:
                # Do not partition because it is fully shadowed
                copied_space = deepcopy(curr_space)
                default = copied_space[param_id].default
                copied_space[param_id].option_expr = "['{0}']".format(default)
                next_queue.append(copied_space)
                log.debug('%d: Stop partition %s due to shadow', ptr, param_id)
            elif not parts or accum_part + len(parts) > limit:
                # Do not partition because it is either
                # 1) not a partitionable parameter, or
                # 2) the accumulated partition number reaches to the limit
                copied_space = deepcopy(curr_space)
                next_queue.append(copied_space)
                log.debug('%d: Stop partition %s due to not partitionable or too many %d', ptr,
                          param_id, limit)
            else:
                # Partition
                for part in parts.values():
                    copied_space = deepcopy(curr_space)
                    copied_space[param_id].option_expr = str(part)
                    copied_space[param_id].default = part[0]
                    next_queue.append(copied_space)
                log.debug('%d: Partition %s to %d parts, so far %d parts', ptr, param_id,
                          len(parts),
                          len(part_queue) + len(next_queue))
        part_queue = next_queue
        ptr += 1
    return [part for part in reversed(part_queue)]
