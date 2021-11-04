"""
The main DSE flow that integrates all modules
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from .config import build_config
from .database import Database, RedisDatabase
from .dsproc.dsproc import compile_design_space, partition
from .dsproc.dsgen import ds_gen
from .evaluator.analyzer import MerlinAnalyzer
from .evaluator.evaluator import BackupMode, Evaluator, MerlinEvaluator
from .evaluator.scheduler import PythonSubprocessScheduler
from .explorer.explorer import AccurateExplorer, FastExplorer
from .logger import get_default_logger
from .parameter import DesignPoint, DesignSpace, get_default_point
from .reporter import Reporter
from .result import HLSResult, Result
from .util import copy_dir


def arg_parser() -> argparse.Namespace:
    """Parse user arguments."""

    parser = argparse.ArgumentParser(description='Automatic Design Space Exploration')
    parser.add_argument('--src-file',
                        required=False,
                        action='store',
                        help='Kernel source code')
    parser.add_argument('--src-dir',
                        required=True,
                        action='store',
                        help='Merlin project directory')
    parser.add_argument('--work-dir',
                        required=True,
                        action='store',
                        default='.',
                        help='DSE working directory')
    parser.add_argument('--config',
                        required=False,
                        action='store',
                        help='path to the configure JSON file')
    parser.add_argument('--db',
                        required=False,
                        action='store',
                        default='',
                        help='path to the result database')
    parser.add_argument('--disable-animation',
                        required=False,
                        action='store_true',
                        default=False,
                        help='disable the animation during the exploration process')
    parser.add_argument(
        '--mode',
        required=False,
        action='store',
        default='fast',
        help='the execution mode (fast-check|complete-check|fast-dse|accurate-dse|fastgen-dse|accurategen-dse)')

    return parser.parse_args()


class Main():
    """The main DSE flow.

    Attributes:
        start_time: Timestamp when launching the flow.
        args: Flow arguments.
        src_dir: Path of the user source project.
        work_dir: Path of the working space.
        out_dir: Path of the output folder.
        eval_dir: Path of the evaluation working space.
        log_dir: Path of logs.
        db_path: Path of persisted database.
        cfg_path: Path of the configuration file.
        log: Logger.
        config: A dictionary of configurations.
        db: Database.
        evaluator: Evaluator.
        reporter: Reporter.
    """

    def __init__(self):
        """Constructor.

        Initialize all necessary modules and working space.
        """
        self.start_time = time.time()
        self.args = arg_parser()

        # Validate check mode
        self.args.mode = self.args.mode.lower()
        if self.args.mode not in ['fast-check', 'complete-check', 'fast-dse', 'accurate-dse', 'fastgen-dse', 'accurategen-dse']:
            print('Error: Invalid mode:', self.args.mode)
            sys.exit(1)

        # Processing path and directory
        self.src_dir = os.path.abspath(self.args.src_dir)
        self.work_dir = os.path.abspath(self.args.work_dir)
        self.out_dir = os.path.join(self.work_dir, 'output')
        self.eval_dir = os.path.join(self.work_dir, 'evaluate')
        self.log_dir = os.path.join(self.work_dir, 'logs')
        if self.args.mode == 'complete-check':
            self.db_path = os.path.join(self.work_dir, 'check.db')
        elif self.args.db:
            self.db_path = os.path.abspath(self.args.db)
        else:
            self.db_path = os.path.join(self.work_dir, 'result.db')
        if not 'gen' in self.args.mode:
            self.cfg_path = os.path.abspath(self.args.config)
            self.src_file = None
        else:
            self.src_file = os.path.abspath(self.args.src_file)

        dir_prefix = os.path.commonprefix([self.src_dir, self.work_dir])
        if dir_prefix in [self.src_dir, self.work_dir]:
            print('Error: Merlin project and workspace cannot be subdirectories!')
            sys.exit(1)
        if not os.path.exists(self.src_dir):
            print('Error: Project folder not found:', self.src_dir)
            sys.exit(1)

        # Initialize workspace
        # Note that the log file must be created after workspace initialization
        # so any message before this point will not be logged.
        bak_dir = self.init_workspace()
        self.log = get_default_logger('Main')
        if bak_dir is not None:
            self.log.warning('Workspace is not empty, backup files to %s', bak_dir)
        self.log.info('Workspace initialized')

        # Generate the config file
        if self.args.mode in ['fastgen-dse', 'accurategen-dse']:
            ds_file = ds_gen(self.src_file)
            if ds_file:
                self.cfg_path = os.path.abspath(ds_file)
            else:
                sys.exit(1)
            
        # Check and load config
        self.config = self.load_config()

        # Stop here if we only need to check the design space definition
        if self.args.mode == 'fast-check':
            self.log.warning('Check mode "FAST": Only check design space syntax and type')
            return

        # Hack the config for check mode:
        # 1) Use gradient algorithm that always evaluates the default point first
        # 2) Set the exploration time to <1 second so it will only explore the default point
        # 3) Use backup error mode in case the checking was failed
        # TODO: Check the bitgen execution
        if self.args.mode == 'complete-check':
            self.log.warning('Check mode "COMPLETE":')
            self.log.warning('1. Check design space syntax and type')
            self.log.warning('2. Evaluate one default point (may take up to 30 mins)')
            self.config['project']['backup'] = 'BACKUP_ERROR'
            self.config['search']['algorithm']['name'] = 'gradient'
            self.config['timeout']['exploration'] = 10e-8

            # We leverage this log to check the evaluation result so it has to be up-to-date
            if os.path.exists('eval.log'):
                os.remove('eval.log')

        # Initialize database
        self.log.info('Initializing the database')
        try:
            self.db = RedisDatabase(self.config['project']['name'], self.db_path)
        except RuntimeError:
            self.log.error('Failed to connect to the database')
            sys.exit(1)
        self.db.load()

        # Initialize evaluator with FAST mode
        self.log.info('Initializing the evaluator')
        self.evaluator = MerlinEvaluator(src_path=self.src_dir,
                                         work_path=self.eval_dir,
                                         db=self.db,
                                         scheduler=PythonSubprocessScheduler(
                                             self.config['evaluate']['worker-per-part']),
                                         analyzer_cls=MerlinAnalyzer,
                                         backup_mode=BackupMode[self.config['project']['backup']],
                                         dse_config=self.config['evaluate'])
        self.evaluator.set_timeout(self.config['timeout'])
        self.evaluator.set_command(self.config['evaluate']['command'])

        # Initialize reporter
        self.reporter = Reporter(self.config, self.db)

        # Compile design space
        self.log.info('Compiling design space for scope map')
        ds = compile_design_space(
            self.config['design-space']['definition'],
            self.evaluator.scope_map if self.args.mode.find('dse') != -1 else None)
        if ds is None:
            self.log.error('Failed to compile design space for scope map')
            return
        curr_point = get_default_point(ds)
        
        if self.args.mode.find('check') == -1:
            self.log.info('Building the scope map')
            if not self.evaluator.build_scope_map(curr_point):
                self.log.error('Failed to build the scope map. See eval.log for details')
                sys.exit(1)

            # Display important configs
            self.reporter.log_config(self.args.mode)

    def init_workspace(self) -> Optional[str]:
        """Initialize the workspace.

        Returns:
            The backup directory if available.
        """

        bak_dir: Optional[str] = None
        try:
            old_files = os.listdir(self.work_dir)
            if old_files:
                bak_dir = tempfile.mkdtemp(prefix='bak_', dir=self.work_dir)

                # Move all files except for config and database files to the backup directory
                for old_file in old_files:
                    # Skip the backup directory of previous runs
                    if old_file.startswith('bak_'):
                        continue
                    full_path = os.path.join(self.work_dir, old_file)
                    if full_path not in [self.cfg_path, self.db_path]:
                        shutil.move(full_path, bak_dir)
                    else:
                        shutil.copy(full_path, bak_dir)
        except FileNotFoundError:
            os.makedirs(self.work_dir)

        return bak_dir

    def load_config(self) -> Dict[str, Any]:
        """Load the DSE configurations.

        Returns:
            A dictionary of configurations.
        """

        try:
            if not os.path.exists(self.cfg_path):
                self.log.error('Config JSON file not found: %s', self.cfg_path)
                raise RuntimeError()

            self.log.info('Loading configurations')
            with open(self.cfg_path, 'r', errors='replace') as filep:
                try:
                    user_config = json.load(filep)
                except ValueError as err:
                    self.log.error('Failed to load config: %s', str(err))
                    raise RuntimeError()

            config = build_config(user_config)
            if config is None:
                self.log.error('Config %s is invalid', self.cfg_path)
                raise RuntimeError()
        except RuntimeError:
            sys.exit(1)

        return config

    def check_eval_log(self) -> None:
        """Parse eval.log and display its errors."""

        error = 0
        if not os.path.exists('eval.log'):
            self.log.error('Evaluation failure: eval.log not found')
        else:
            log_msgs: Set[str] = set()
            with open('eval.log', 'r', errors='replace') as filep:
                for line in filep:
                    if line.find('ERROR') != -1:
                        msg = line[line.find(':') + 2:-1]
                        if msg not in log_msgs:
                            self.log.error(msg)
                            log_msgs.add(msg)
                            error += 1
            if error > 0:
                self.log.error(
                    'The default point encounters %d errors. See %s/evaluate for details', error,
                    self.args.work_dir)

    def gen_fast_outputs(self) -> List[DesignPoint]:
        """Generate outputs after fast mode.

        Note:
            The best cache in the DB will be cleaned up by this function.

        Returns:
            A list of design points output by fast mode.
        """

        def geomean(seq):
            """A simple function to compute geometric mean of a list"""
            return math.exp(math.fsum(math.log(x) if x > 0 else 0 for x in seq) / len(seq))

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        out_fast_dir = os.path.join(self.out_dir, 'fast')

        # Clean output directory
        if os.path.exists(out_fast_dir):
            shutil.rmtree(out_fast_dir)
        os.makedirs(out_fast_dir)

        points = []
        output = []
        idx = 0

        # Sort the results by 1) quality and 2) geomean of resource util and take the first N
        results: List[Result] = [
            r for r in self.db.query_all()
            if isinstance(r, HLSResult) and r.valid and r.ret_code != Result.RetCode.DUPLICATED
        ]
   
        results.sort(key=lambda r: (r.quality, 1.0 / geomean(
            [v for k, v in r.res_util.items() if k.startswith('util')])),
                     reverse=True)
        results = results[:int(self.config['project']['fast-output-num'])]
        for result in results:
            job = self.evaluator.create_job()
            if not job:
                continue

            assert result.point is not None
            self.evaluator.apply_design_point(job, result.point)
            points.append(result.point)
            os.rename(job.path, os.path.join(out_fast_dir, str(idx)))
            result.path = str(idx)
            output.append(result)
            idx += 1

        rpt = self.reporter.report_output(output)
        if rpt:
            with open(os.path.join(out_fast_dir, 'output.rpt'), 'w') as filep:
                filep.write(rpt)

        # Draw result distribution with Pareto curve
        self.reporter.draw_pareto_curve(os.path.join(out_fast_dir, 'result_dist.pdf'))

        return points

    def launch_fast(self, ds_list: List[DesignSpace]) -> List[DesignPoint]:
        """Launch fast exploration.

        Args:
            ds_list: A list of design space partitions.

        Returns:
            A list of output design points.
        """

        pool = []

        # Launch a thread pool
        with ThreadPoolExecutor(max_workers=len(ds_list)) as executor:
            for idx, ds in enumerate(ds_list):
                pool.append(
                    executor.submit(self.fast_runner,
                                    tag='part{0}'.format(idx),
                                    ds=ds,
                                    db=self.db,
                                    evaluator=self.evaluator,
                                    config=self.config))

            self.log.info('%d explorers have been launched', len(pool))

            timer: float = (time.time() - self.start_time) / 60.0  # in minutes
            while any([not exe.done() for exe in pool]):
                time.sleep(1)
                # Only keep the best result
                while self.db.best_cache.qsize() > 1:
                    self.db.best_cache.get()

                # Print animation to let user know we are still working, or print dots every
                # 5 mins if user disables the animation.
                if self.args.disable_animation:
                    if int(timer) % 5 == 0:
                        print('.')
                else:
                    self.reporter.log_best()

                    count = 0
                    for idx in range(len(ds_list)):
                        part_cnt = self.db.query('meta-expr-cnt-{0}'.format('part{0}'.format(idx)))
                        if part_cnt:
                            try:
                                count += int(part_cnt)
                            except ValueError:
                                pass
                    self.reporter.print_status(timer, count)
                timer += 0.0167

        if self.args.mode == 'complete-check':
            return []

        # Backup database
        self.db.persist()

        # Report and summary
        summary, detail = self.reporter.report_summary()
        for line in summary.split('\n'):
            if line:
                self.log.info(line)
        with open(os.path.join(self.work_dir, 'summary_fast.rpt'), 'w') as filep:
            filep.write(summary)
            filep.write('\n\n')
            filep.write(detail)

        # Create outputs
        points = self.gen_fast_outputs()
        self.log.info('Outputs of fast exploration are generated')
        return points

    @staticmethod
    def fast_runner(tag: str, ds: DesignSpace, db: Database, evaluator: Evaluator,
                    config: Dict[str, Any]) -> None:
        """Perform fast DSE for a given design space.

        This is a static method that is supposed to be used for forking explorer threads.

        Args:
            tag: A tag of this run.
            ds: Design space.
            db: Database.
            evaluator: Evaluator.
            config: Configuration.
        """

        explorer = FastExplorer(ds=ds,
                                db=db,
                                evaluator=evaluator,
                                timeout=config['timeout']['exploration'],
                                tag=tag)
        try:
            explorer.run(config['search']['algorithm'])
        except Exception as err:  # pylint:disable=broad-except
            log = get_default_logger('DSE')
            log.error('Encounter error during the fast exploration: %s', str(err))
            log.error(traceback.format_exc())

    def gen_accurate_outputs(self) -> None:
        """Generate final outputs."""

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        out_accurate_dir = os.path.join(self.out_dir, 'accurate')

        # Clean output directory
        if os.path.exists(out_accurate_dir):
            shutil.rmtree(out_accurate_dir)
        os.makedirs(out_accurate_dir)

        best_path = ''
        best_quality = -float('inf')
        output = []
        idx = 0

        # Fetch accurate results and sort by quality
        keys: List[str] = [k for k in self.db.query_keys() if k.startswith('lv3:')]
        results: List[Result] = [r for r in self.db.batch_query(keys)]
        for result in results:
            assert result.point is not None
            assert result.path is not None
            output_path = os.path.join(out_accurate_dir, str(idx))
            copy_dir(result.path, output_path)
            result.path = str(idx)
            output.append(result)
            if best_quality == -float('inf') or result.quality > best_quality:
                best_quality = result.quality

                # The relative path for creating a symbolic link
                best_path = os.path.join('./accurate', str(idx))
            idx += 1

        rpt = self.reporter.report_output(output)
        if rpt:
            with open(os.path.join(out_accurate_dir, 'output.rpt'), 'w') as filep:
                filep.write(rpt)

            # Make a symbolic link for the best one
            if best_path:
                os.symlink(best_path, os.path.join(self.out_dir, 'best'))

        # Draw result distribution with Pareto curve
        self.reporter.draw_pareto_curve(os.path.join(out_accurate_dir, 'result_dist.pdf'), True)

    def launch_accurate(self, points: List[DesignPoint]) -> None:
        """Launch accurate exploration.

        Args:
            points: A list of target design poitns.
        """

        # Enforce backup all since this process is very time-consuming
        if self.evaluator.backup_mode != BackupMode.BACKUP_ALL:
            self.log.info('Backup mode is set to ALL to keep all P&R results')
            self.evaluator.backup_mode = BackupMode.BACKUP_ALL

        with ThreadPoolExecutor(max_workers=1) as executor:
            proc = executor.submit(self.accurate_runner,
                                   points=points,
                                   db=self.db,
                                   evaluator=self.evaluator,
                                   config=self.config)

            timer: float = (time.time() - self.start_time) / 60.0  # in minutes
            while not proc.done():
                time.sleep(1)
                count = self.db.query('meta-expr-cnt-accurate')
                try:
                    self.reporter.print_status(timer, int(count), 2)
                except (TypeError, ValueError):
                    self.reporter.print_status(timer, 0, 2)
                timer += 0.0167

        # Backup database again
        self.db.persist()

        summary, detail = self.reporter.report_summary()
        with open(os.path.join(self.work_dir, 'summary_accurate.rpt'), 'w') as filep:
            filep.write(summary)
            filep.write('\n\n')
            filep.write(detail)

        # Create outputs
        self.gen_accurate_outputs()
        self.log.info('The outputs is generated. See output/accuurate/output.rpt for details')

    @staticmethod
    def accurate_runner(points: List[DesignPoint], db: Database, evaluator: Evaluator,
                        config: Dict[str, Any]):
        """Perform phase 2 DSE for a given set of points.

        This is a static method that is supposed to be used for forking explorer threads.

        Args:
            points: A list of target design poitns.
            db: Database.
            evaluator: Evaluator.
            config: Configuration.
        """

        explorer = AccurateExplorer(points=points, db=db, evaluator=evaluator, tag='accurate')

        try:
            explorer.run(config['search']['algorithm'])
        except Exception as err:  # pylint:disable=broad-except
            log = get_default_logger('DSE')
            log.error('Encounter error during the accurate exploration: %s', str(err))
            log.error(traceback.format_exc())

    def main(self) -> None:
        """The main function of the DSE flow."""
        # Compile design space
        self.log.info('Compiling design space')
        ds = compile_design_space(
            self.config['design-space']['definition'],
            self.evaluator.scope_map if self.args.mode.find('dse') != -1 else None)
        if ds is None:
            self.log.error('Failed to compile design space')
            return

        # Partition design space
        self.log.info('Partitioning the design space to at maximum %d parts',
                      int(self.config['design-space']['max-part-num']))
        ds_list = partition(ds, int(self.config['design-space']['max-part-num']))
        if ds_list is None:
            self.log.error('No design space partition is available for exploration')
            return

        #with open('ds_part{0}.json'.format(idx), 'w') as filep:
        #    filep.write(
        #        json.dumps({n: p.__dict__
        #                    for n, p in ds.items()}, sort_keys=True, indent=4))

        self.log.info('%d parts generated', len(ds_list))

        if self.args.mode == 'fast-check':
            self.log.info('Finish checking the design space (fast mode)')
            return

        # TODO: profiling and pruning

        # Launch exploration
        try:
            if self.args.mode.find('dse') != -1:
                self.log.info('Start the exploration')
            fast_points = self.launch_fast(ds_list)
        except KeyboardInterrupt:
            pass

        if self.args.mode == 'complete-check':
            self.check_eval_log()
            self.log.info('Finish checking the design space (complete mode)')
            return

        if self.args.mode == 'fast-dse' or self.args.mode == 'fastgen-dse':
            self.log.info('Finish the exploration')
        else:
            self.log.info('Finish the phase 1 exploration with %d candidates', len(fast_points))
            self.log.info('Start the phase 2 exploration')

            # Run phase 2 in ACCURATE mode
            try:
                self.launch_accurate(fast_points)
            except KeyboardInterrupt:
                pass

        # Backup logs
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir)

        for log in glob.glob('*.log'):
            shutil.move(log, os.path.join(self.log_dir, log))
