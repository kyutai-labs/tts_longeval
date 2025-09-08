# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Code for the `Runner`, which runs jobs with GPUs, either locally
or through SLURM.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError, Future
from contextlib import ExitStack
import logging
import math
from pathlib import Path
import subprocess as sp
import time

from pydantic import BaseModel
import submitit

from . import task


logger = logging.getLogger(__name__)


class SubmititConfig(BaseModel):
    """
    Config passed to `submitit`.

    Args:
        slurm: if true, uses SLURMM, otherwise detects the local GPUs.
            Most of the params after have no effect if `slurm` is false.
        partition: slurm partition to use.
        max_gpus: maximum number of GPUs to use in total.
        partition: SLURM partition to use.
        max_gpus: maximum number of GPUs to use in total.
        mem_per_gpu: memory in GB per GPU.
        cpus_per_gpu: number of CPUs per GPU.
        time: maximum job duration in minutes.
    """
    slurm: bool = True
    partition: str = 'default'
    max_gpus: int = 64
    gpus_per_task: int = 8
    mem_per_gpu: int | None = 50
    cpus_per_gpu: int = 8
    time: int = 1440


class RunnerConfig(BaseModel):
    """Config for the `Runner` in the config file.

    Args:
        submitit: config for submitit.
        threads: number of local CPU threads, used only for calling API based TTS.

    """
    submitit: SubmititConfig
    threads: int = 4

    def get(self, submitit_folder: Path, debug: bool) -> 'Runner':
        return Runner(submitit_folder, self.submitit, self.threads, debug)


class Runner:
    """
    Run jobs and monitor them.
    Args:
        submitit_folder: where to store logs and stuff.
        submitit_config: config for submitit.
        threads: number of local threads.
        debug: if True, will end if any of the jobs crashed.
        log_every: how often to print the number of jobs running and status.
    """
    def __init__(self, submitit_folder: Path, submitit_config: SubmititConfig, threads: int = 4,
                 debug: bool = False, log_every: float = 60.):
        self.submitit_config = submitit_config
        self.threads = threads
        self.debug = debug
        self.log_every = log_every
        self.pool = ThreadPoolExecutor(threads)
        if submitit_config.slurm:
            self.executor = submitit.SlurmExecutor(folder=submitit_folder, max_num_timeout=3)
        else:
            self.executor = submitit.LocalExecutor(folder=submitit_folder)
        self._update_executor_parameters()
        self.executor.update_parameters(
            job_name='tts_longeval_dispatcher',
            stderr_to_stdout=True)
        self.stack = ExitStack()

    def _update_executor_parameters(self):
        conf = self.submitit_config
        gpus = conf.gpus_per_task
        kwargs = {}
        if conf.slurm:
            if conf.mem_per_gpu:
                mem = conf.mem_per_gpu * gpus
                kwargs['mem'] = f"{mem}GB"
            kwargs['gres'] = f'gpu:{gpus}'
            kwargs['ntasks_per_node'] = 1
            kwargs['cpus_per_task'] = gpus * conf.cpus_per_gpu
            kwargs['partition'] = conf.partition
            kwargs['time'] = conf.time
            kwargs['exclude'] = 'par2dc5-ai-prd-cl02s04dgx31'
        else:
            kwargs['gpus_per_node'] = conf.gpus_per_task
            kwargs['timeout_min'] = conf.time
        self.executor.update_parameters(**kwargs)

    def _run_gpu(self, multi_tasker_gpu: task.MultiTasker, cpu_jobs: list[Future]):
        with ExitStack() as stack:
            num_jobs = int(math.ceil(self.submitit_config.max_gpus / self.submitit_config.gpus_per_task))
            jobs = []
            with ExitStack() as maybe_batch:
                if self.submitit_config.slurm:
                    maybe_batch.enter_context(self.executor.batch())
                for job_idx in range(num_jobs):
                    if not self.submitit_config.slurm:
                        gpus_per_task = self.submitit_config.gpus_per_task
                        visible_gpus = list(range(job_idx * gpus_per_task, (job_idx + 1) * gpus_per_task))
                        self.executor.update_parameters(visible_gpus=visible_gpus)
                    jobs.append(self.executor.submit(multi_tasker_gpu.run))

            for job in jobs:
                stack.callback(job.cancel)
            jobid = jobs[0].job_id.rsplit('_', 1)[0]
            logger.info('Runner: job id: %s', jobid)
            log = self.executor.folder / (jobs[0].job_id + '_0_log.out')
            logger.info('Runner: first log: %s', log)
            if self.debug:
                proc = sp.Popen(['tail', '-F', log])
                stack.callback(proc.terminate)
            known_failures = set()
            last_log_time = time.time()
            while True:
                if self.submitit_config.slurm:
                    jobs[0].watcher.update()
                done = 0
                failed = 0
                for idx, job in enumerate(jobs):
                    if job.done():
                        if job.state in ['RUNNING', 'PENDING']:
                            continue
                        elif job.state in ['COMPLETED', 'FINISHED']:
                            done += 1
                        else:
                            failed += 1
                            if idx > 1 or not self.debug and idx not in known_failures:
                                known_failures.add(idx)
                                stdout = job.stdout()
                                if stdout is None:
                                    logger.error("Runner: no stdout for %s with state %s", jobid, job.state)
                                else:
                                    stdout = '\n'.join(stdout.split('\n')[-100:])
                                    logger.error("Runner: stdout for %s with state %s:\n %s", jobid, job.state, stdout)

                for cpu_job in cpu_jobs:
                    # Let's try to get the tracebacks early on.
                    try:
                        cpu_job.result(timeout=0)
                    except TimeoutError:
                        continue

                if time.time() - last_log_time > self.log_every:
                    logger.info(f"Runner: {done: 4d} done, {failed: 4d} failed / {len(jobs): 4d} jobs.")
                    last_log_time = time.time()
                if self.debug and failed:
                    raise RuntimeError("One job failed.")
                if done + failed == len(jobs):
                    break
                time.sleep(10.)

    def run(self, multi_tasker_gpu: task.MultiTasker, multi_tasker_cpu: task.MultiTasker):
        with self.pool:
            cpu_jobs = []
            if multi_tasker_cpu:
                for _ in range(self.threads):
                    cpu_jobs.append(self.pool.submit(multi_tasker_cpu.run))

            if multi_tasker_gpu:
                self._run_gpu(multi_tasker_gpu, cpu_jobs)

            for cpu_job in cpu_jobs:
                cpu_job.result()
