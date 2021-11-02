"""
The main module of job schedulers.
"""
import glob
import os
import shutil
import signal
import time
from math import ceil
from typing import List, Tuple, Optional

from ..logger import get_eval_logger
from ..result import Job, Result
from ..util import copy_dir


class Scheduler():
    """The base class of job scheduler.

    Attributes:
        log: The logger.
        max_worker: The maximum number of allowed workers.
    """

    def __init__(self, max_worker: int = 8):
        self.log = get_eval_logger('Scheduler')
        self.max_worker = max_worker

    def run(self, jobs: List[Job], keep_files: List[str], cmd: str,
            timeout: Optional[int] = None) -> List[Tuple[str, Result.RetCode]]:
        """The main API of scheduling and running given jobs.

        Args:
            job: A list of job objects to be scheduled.
            keep_files: A list of file name (support wildcards) to indicate which files
                        should be kept for result analysis.
            cmd: A string of command for execution. Note that we may extend this part
                 to another evaluation function instead of a single string in the future.
            timeout: The timeout in minutes of the evaluation. None means no timeout.

        Returns:
            A list of each job key and its corresponding return code.
        """
        raise NotImplementedError()


class PythonSubprocessScheduler(Scheduler):
    """The scheduler implementation using Python subprocess."""

    @staticmethod
    def backup_files_and_rmtree(src_path: str,
                                dst_path: str,
                                file_list: Optional[List[str]] = None) -> None:
        """Backup files from working directory to the job directory and remove working directory.

        Args:
            src_path: The working directory.
            dst_path: The job directory.
            file_list: A list of files we want to keep. None means keep all.

        Returns:
            This function is slient and will not check if the backup was success or not.
        """
        log = get_eval_logger('Scheduler')
        if not file_list:
            shutil.rmtree(dst_path)
            shutil.move(src_path, dst_path)
        else:
            for file_expr in file_list:
                # Prepare destination folder
                dst_full_path = os.path.dirname(os.path.join(dst_path, file_expr))
                if not os.path.exists(dst_full_path):
                    os.makedirs(dst_full_path)

                # Walk through all match files under the folder
                for file_path in glob.glob(os.path.join(src_path, file_expr)):
                    file_name = os.path.basename(file_path)
                    try:
                        dst_file = os.path.join(dst_full_path, file_name)
                        shutil.move(file_path, dst_file)
                    except FileNotFoundError as err:
                        log.info('Failed to move %s to %s: %s', file_path, dst_path, str(err))
            shutil.rmtree(src_path)

    def run(self, jobs: List[Job], keep_files: List[str], cmd: str,
            timeout: Optional[int] = None) -> List[Tuple[str, Result.RetCode]]:
        #pylint: disable=missing-docstring

        from subprocess import Popen, DEVNULL

        rets = {job.key: Result.RetCode.UNAVAILABLE for job in jobs}

        # Batch jobs when the number is larger than the max workers
        num_batch = ceil(len(jobs) / self.max_worker)
        for batch in range(num_batch):
            procs = []
            for offset in range(self.max_worker):
                idx = batch * self.max_worker + offset
                if idx >= len(jobs):
                    break
                copy_dir(jobs[idx].path, '{0}_work'.format(jobs[idx].path))

                # Since we use shell=True to launch a new bash in order to make sure the command
                # is executed as it in the bash shell, we need to also set start_new_session=True
                # in order to send the kill signal when timeout or interrupt because proc.kill()
                # is not working when shell=True.
                # See https://stackoverflow.com/questions/4789837 for details.
                proc = Popen('cd {0}_work; {1}'.format(jobs[idx].path, cmd),
                             stdout=DEVNULL,
                             stderr=DEVNULL,
                             shell=True,
                             start_new_session=True)
                procs.append((idx, proc))

            if not procs:
                break

            time_limit = float('inf') if timeout is None else timeout
            self.log.info('Launching batch %d/%d with %d jobs and timeout %.2f mins', batch + 1,
                          num_batch, len(procs), time_limit)
            timer = time.time()
            try:
                while (time.time() - timer) < time_limit * 60.0 and procs:
                    old_procs = list(procs)
                    procs = []
                    for idx, proc in old_procs:
                        ret = proc.poll()
                        # Finished, check if success, remove from list, and backup wanted files
                        if ret is not None and ret != 0:
                            self.log.error('Command "%s" has non-zero exit code: %d', cmd, ret)
                            self.backup_files_and_rmtree('{0}_work'.format(jobs[idx].path),
                                                         jobs[idx].path)
                        elif ret is not None:
                            rets[jobs[idx].key] = Result.RetCode.PASS
                            self.backup_files_and_rmtree('{0}_work'.format(jobs[idx].path),
                                                         jobs[idx].path, keep_files)
                        else:
                            # Still running
                            procs.append((idx, proc))
                    time.sleep(1)

                if procs:
                    # One or more processes are timeout.
                    # Note that timeout is considered as a success run
                    self.log.info('%d processes timeout (%.2f mins)', len(procs), time_limit)
                    for idx, proc in procs:
                        rets[jobs[idx].key] = Result.RetCode.TIMEOUT
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except KeyboardInterrupt:
                self.log.warning('Received user keyboard interrupt, stopping the process.')
                for idx, proc in procs:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                break

        return list(rets.items())
