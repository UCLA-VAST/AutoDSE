"""
The main module of explorer.
"""
import time
from typing import Any, Dict, Generator, List, Optional

from .algorithmfactory import AlgorithmFactory
from ..evaluator.evaluator import Evaluator
from ..logger import get_algo_logger
from ..database import Database
from ..parameter import DesignPoint, DesignSpace, gen_key_from_design_point
from ..result import Job, Result


class Explorer():
    """Main explorer class.

    Attributes:
        db: The result database.
        evaluator: The evaluator.
        tag: The string tag of this explorer.
        algo_log_file_name: The prefix name of log file.
        log: Logger object.
        best_result: So far best result.
        explored_point: So far explored point.
    """

    def __init__(self, db: Database, evaluator: Evaluator, tag: str):
        """Constructor.

        Args:
            db: Database.
            evaluator: Evaluator.
            tag: A unique tag.
        """
        self.db = db
        self.evaluator = evaluator
        self.tag = tag
        self.algo_log_file_name = '{0}_algo.log'.format(self.tag)
        self.log = get_algo_logger('Explorer', '{0}_expr.log'.format(self.tag))

        # Status checking
        self.best_result: Result = Result()
        self.explored_point = 0

    def create_job_and_apply_point(self, point) -> Optional[Job]:
        """Create a new job and apply the given design point.

        Args:
            point: The point to be applied.

        Returns:
            The created job, or None if failed.
        """

        job = self.evaluator.create_job()
        if job:
            if not self.evaluator.apply_design_point(job, point):
                return None
        else:
            self.log.error('Fail to create a new job (disk space?)')
            return None
        return job

    def update_best(self, result: Result) -> None:
        """Keep tracking the best result found in this explorer.

        Args:
            result: The new result to be checked.

        """
        if result.valid and result.quality > self.best_result.quality:
            self.best_result = result
            self.log.info('Found a better result at #%04d: Quality %.1e, Perf %.1e',
                          self.explored_point, result.quality, result.perf)

    def run(self, algo_config: Dict[str, Any]) -> None:
        """The main function of the explorer to launch the search algorithm.

        Args:
            algo_name: The corresponding algorithm name for running this exploration.
            algo_config: The configurable values for the algorithm.
        """
        raise NotImplementedError()


class FastExplorer(Explorer):
    """
    Fast explorer uses serach algorithm to find the best point in a large design space. For each
    explored point, it runs level 1 and level 2 evaluation to get an estimated QoR in a relative
    short time.

    Attributes:
        timeout: Timeout in seconds for each job evaluation.
        ds: The given design space.
    """

    def __init__(self, db: Database, evaluator: Evaluator, timeout: int, tag: str,
                 ds: DesignSpace):
        """Constructor.

        Args:
            db: Database.
            evaluator: Evaluator.
            timeout: Timeout of each job evaluation in minutes.
            tag: A unique tag.
            ds: Design space.
        """
        super(FastExplorer, self).__init__(db, evaluator, tag)
        self.timeout = timeout * 60.0
        self.ds = ds

    def run(self, algo_config: Dict[str, Any]) -> None:
        #pylint:disable=missing-docstring

        # Create a search algorithm generator
        algo = AlgorithmFactory.make(algo_config, self.ds, self.algo_log_file_name)
        gen_next = algo.gen()

        timer = time.time()
        duplicated_iters = 0
        results: Optional[Dict[str, Result]] = None
        while (time.time() - timer) < self.timeout:
            try:
                # Generate the next set of design points
                next_points = gen_next.send(results)
                self.log.debug('The algorithm generates %d design points', len(next_points))
            except StopIteration:
                break

            results = {}

            # Create jobs and check duplications
            jobs: List[Job] = []
            keys: List[str] = []
            lv2_keys = [
                'lv2:{0}'.format(gen_key_from_design_point(point)) for point in next_points
            ]
            for point, result in zip(next_points, self.db.batch_query(lv2_keys)):
                key = gen_key_from_design_point(point)
                if result is not None:
                    # Already has HLS result, skip
                    self.update_best(result)
                    results[key] = result
                else:
                    # Miss HLS result, further check level 1
                    keys.append(key)

            lv1_keys = ['lv1:{0}'.format(key) for key in keys]
            for point, result in zip(next_points, self.db.batch_query(lv1_keys)):
                key = gen_key_from_design_point(point)
                if result is not None:
                    # Already has Merlin result, skip
                    self.update_best(result)
                    results[key] = result
                else:
                    # No result in the DB, generate
                    job = self.create_job_and_apply_point(point)
                    if job:
                        jobs.append(job)
                    else:
                        return
            if not jobs:
                duplicated_iters += 1
                self.log.debug('All design points are already evaluated (%d iterations)',
                               duplicated_iters)
                continue

            duplicated_iters = 0

            # Evaluate design points using level 1 to fast check if it is suitable for HLS
            self.log.debug('Evaluating %d design points: Level 1', len(jobs))
            pending: List[Job] = []
            for key, result in self.evaluator.submit(jobs, 1):
                if result.ret_code == Result.RetCode.PASS:
                    job = self.create_job_and_apply_point(result.point)
                    if job:
                        pending.append(job)
                    else:
                        return
                else:
                    results[key] = result

            if pending:
                # Evaluate design points using level 2 that runs HLS
                self.log.debug('Evaluating %d design points: Level 2' % (len(pending)))
                for key, result in self.evaluator.submit(pending, 2):
                    self.update_best(result)
                    results[key] = result
            else:
                self.log.info('All points are stopped at level 1')

            self.explored_point += len(jobs)
            self.db.commit('meta-expr-cnt-{0}'.format(self.tag), self.explored_point)

        self.log.info('Explored %d points', self.explored_point)


class AccurateExplorer(Explorer):
    """
    Currently we simply evaluate all given points and mark the best one. The future opportunities
    here could be an algorithm for tuning design tool parameters.

    Attributes:
        points: A set of points to be explored.
    """

    def __init__(self, db: Database, evaluator: Evaluator, tag: str, points: List[DesignPoint]):
        """Constructor.

        Args:
            db: Database.
            evaluator: Evaluator.
            timeout: Timeout of each job evaluation in minutes.
            tag: A unique tag.
            points: A set of points to be explored.
        """
        super(AccurateExplorer, self).__init__(db, evaluator, tag)
        self.points = points

    def run(self, algo_config: Dict[str, Any]) -> None:
        #pylint:disable=missing-docstring

        def chunk(points: List[DesignPoint],
                  size: int) -> Generator[List[DesignPoint], None, None]:
            """Chunk design point list to the given size.

            Args:
                points: A full list of design points.
                size: The maximum size of each chunk.

            Returns:
                A generator to chunk the list.
            """

            for i in range(0, len(points), size):
                yield points[i:i + size]

        batch_size = int(algo_config['exhaustive']['batch-size'])
        self.log.info('Batch size is set to %d', batch_size)

        for points in list(chunk(self.points, batch_size)):
            # Create jobs
            jobs: List[Job] = []
            for point in points:
                job = self.create_job_and_apply_point(point)
                if job:
                    jobs.append(job)
                else:
                    return

            # Evaluate design points
            self.log.info('Evaluating %d design points: Level 3', len(jobs))
            for _, result in self.evaluator.submit(jobs, 3):
                self.update_best(result)

            self.explored_point += len(jobs)
            self.db.commit('meta-expr-cnt-accurate', self.explored_point)

        self.log.info('Explored %d points', self.explored_point)
