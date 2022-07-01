"""
Run a single evaluation on a design point
"""
from typing import Any, Dict, Generator, List, Optional
import json
import pickle
from os.path import join
import argparse

from .explorer import Explorer
from ..parameter import DesignPoint, DesignSpace, gen_key_from_design_point
from ..result import HLSResult, Job, MerlinResult, Result
from ..evaluator.evaluator import Evaluator
from ..evaluator.analyzer import MerlinAnalyzer
from ..evaluator.evaluator import MerlinEvaluator
from ..database import RedisDatabase
from ..evaluator.scheduler import PythonSubprocessScheduler

def run_query(point, src_dir, eval_dir, kernel, config_path = None, timeout = 200) -> Optional[Dict[str, Result]]:
    key = gen_key_from_design_point(point)
    print(key)
    
    try:
        db = RedisDatabase("result")
    except RuntimeError:
        print('Failed to connect to the database')
    db.load()

    def load_config(config_path) -> Dict[str, Any]:
        if config_path:
            pass
        else:
            config_path = join(src_dir, kernel, "ds_info.json")
        with open(config_path, 'r', errors='replace') as filep:
            try:
                user_config = json.load(filep)
            except ValueError as err:
                print('Failed to load config: %s', str(err))
                raise RuntimeError()

        # Build config
        config: Dict[str, Any] = {}
        for key, attr in user_config.items():
            curr = config
            levels = key.split('.')
            for level in levels[:-1]:
                if level not in curr:
                    curr[level] = {}
                curr = curr[level]
            curr[levels[-1]] = attr
        if config is None:
            print('Config %s is invalid', config_path)
            raise RuntimeError()
        config['timeout']['hls'] = int(timeout)
        return config

    def create_job_and_apply_point(point, evaluator) -> Optional[Job]:
        """Create a new job and apply the given design point.

        Args:
            point: The point to be applied.

        Returns:
            The created job, or None if failed.
        """

        job = evaluator.create_job()
        if job:
            if not evaluator.apply_design_point(job, point):
                return None
        else:
            print('Fail to create a new job (disk space?)')
            return None
        return job

    config = load_config(config_path)
    evaluator = MerlinEvaluator(src_path=src_dir,
                                work_path=eval_dir,
                                db=db,
                                scheduler=PythonSubprocessScheduler(1),
                                analyzer_cls=MerlinAnalyzer,
                                backup_mode="BACKUP_ERROR",
                                dse_config=config['evaluate'])
    evaluator.set_command(config['evaluate']['command'])
    evaluator.set_timeout(config['timeout'])

    results: Optional[Dict[str, Result]] = None
    results = {}

    jobs = create_job_and_apply_point(point, evaluator)

    pending: List[Job] = []
    for key, result in evaluator.submit([jobs], 1):
        print(result.ret_code)
        if result.ret_code == Result.RetCode.PASS or result.ret_code == Result.RetCode.EARLY_REJECT:
            job = create_job_and_apply_point(result.point, evaluator)
            if job:
                pending.append(job)
            else:
                break
        else:
            results['lv1:'+key] = result

    if pending:
        # Evaluate design points using level 2 that runs HLS
        print('Evaluating %d design points: Level 2' % (len(pending)))
        for key, result in evaluator.submit(pending, 2):
            #print(key, result)
            #self.update_best(result)
            results['lv2:'+key] = result
    else:
        print('All points are stopped at level 1')

    return results

def kernel_parser() -> argparse.Namespace:
    """Parse user arguments."""

    parser = argparse.ArgumentParser(description='Running Queries')
    parser.add_argument('--kernel',
                        required=True,
                        action='store',
                        help='Kernel Name')
    parser.add_argument('--config',
                        required=True,
                        action='store',
                        help='Config File')
    parser.add_argument('--src-dir',
                        required=True,
                        action='store',
                        default='.',
                        help='Source Directory')
    parser.add_argument('--work-dir',
                        required=True,
                        action='store',
                        default='.',
                        help='work Directory')
    parser.add_argument('--id',
                        required=False,
                        action='store',
                        default='0',
                        help='the ID of design in the batch')
    parser.add_argument('--timeout',
                        required=False,
                        action='store',
                        default='200',
                        help='timeout for running HLS synthesis')
    # parser.add_argument('--point',
    #                     required=True,
    #                     action='store',
    #                     type=str,
    #                     help='Design Point')

    return parser.parse_args()
    
args = kernel_parser()

point = pickle.load(open(f'./localdse/kernel_results/{args.kernel}_point_{args.id}.pickle', 'rb'))
print(point)

q_result = run_query(point, args.src_dir, args.work_dir, args.kernel, args.config, args.timeout)
with open(f'./localdse/kernel_results/{args.kernel}_{args.id}.pickle', 'wb') as handle:
    pickle.dump(q_result, handle, protocol=pickle.HIGHEST_PROTOCOL)




