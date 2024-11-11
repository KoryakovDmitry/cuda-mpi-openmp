import asyncio
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Any, Dict

import numpy as np
import pandas as pd

TIME_KERNEL_EXE_PATTERN = r"execution time: <([\d.]+) ms>"


async def get_time_kernel_exe(
    text: str, pattern: str = TIME_KERNEL_EXE_PATTERN
) -> Optional[float]:
    match = re.search(pattern, text)
    if match:
        time_ms = float(match.group(1))
        return time_ms
    return None


async def print_stat_time(data_np):
    mean = np.mean(data_np)
    median = np.median(data_np)
    minimum = np.min(data_np)
    maximum = np.max(data_np)
    std_dev = np.std(data_np)

    print(f"Mean: {mean} ms")
    print(f"Median: {median} ms")
    print(f"Min: {minimum} ms")
    print(f"Max: {maximum} ms")
    print(f"Standard Deviation: {std_dev} ms")


@dataclass
class TaskResult:
    test_verification_result: Optional[bool]
    task_result: Optional[Any]
    time_kernel_exe_ms: Optional[float]


@dataclass
class SubProcessResult(TaskResult):
    status: bool
    err: Optional[str]


class BaseLabProcessor:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)

    async def get_data_state(
        self,
    ) -> Dict[str, Any]:
        pass

    async def pre_process(self, **kwargs) -> str:
        pass

    async def verify_result(self, task_result: Any) -> bool:
        pass

    async def get_task_result(self, task_result_string: str) -> Any:
        pass

    async def post_process(self, result_stdout: str) -> TaskResult:
        result_stdout_split = result_stdout.split("\n")
        time_kernel_exe_ms_task = asyncio.create_task(
            get_time_kernel_exe(result_stdout_split[0])
        )

        task_result_string = "\n".join(result_stdout_split[1:])
        task_result = (await self.get_task_result(task_result_string),)
        test_verification_result = await self.verify_result(task_result)
        return TaskResult(
            time_kernel_exe_ms=await time_kernel_exe_ms_task,
            test_verification_result=test_verification_result,
            task_result=task_result,
        )


async def run_subprocess(
    binary_path: str,
    lab_processor: BaseLabProcessor,
    kernel_size_1: Optional[int] = None,
    kernel_size_2: Optional[int] = None,
) -> SubProcessResult:
    try:
        input_str = await lab_processor.pre_process()
        if kernel_size_1 and kernel_size_2:
            input_str = f"{kernel_size_1}\n{kernel_size_2}\n{input_str}"

        result = subprocess.run(
            [binary_path],
            input=input_str,
            text=True,
            capture_output=True,
            check=True,
        )
        task_result: TaskResult = await lab_processor.post_process(
            result_stdout=result.stdout
        )

        return SubProcessResult(
            test_verification_result=task_result.test_verification_result,
            task_result=task_result.task_result,
            time_kernel_exe_ms=task_result.time_kernel_exe_ms,
            status=True,
            err=None,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr.strip()
        return SubProcessResult(
            test_verification_result=None,
            task_result=None,
            time_kernel_exe_ms=None,
            status=False,
            err=err,
        )


class BaseTester:
    def __init__(
        self,
        binary_path_cuda: str,
        k_times: int,
        kernel_sizes: List[List[Optional[int]]],
        binary_path_cpu: Optional[str] = None,
    ):
        self.binary_path_cuda = binary_path_cuda
        self.binary_path_cpu = binary_path_cpu
        self.kernel_sizes = kernel_sizes
        self.k_times = k_times

    async def run_experiment(
        self,
        binary_path: str,
        kernel_sizes: List[List[Optional[int]]],
        lab_processor: BaseLabProcessor,
    ) -> Optional[pd.DataFrame]:
        bin_name = os.path.splitext(os.path.basename(binary_path))[0]
        dir2save = os.path.dirname(binary_path)

        print(f"[Experiment bin_name=<{bin_name}>] START")
        df_scores = None
        tasks = []

        for i in range(self.k_times):
            print(f"[Experiment bin_name=<{bin_name}> task={i}] started")

            for kernel_size_1, kernel_size_2 in kernel_sizes:
                tasks.append(
                    {
                        "idx_run_time": i,
                        "bin_name": bin_name,
                        "time_st": time.time(),
                        "task": asyncio.create_task(
                            run_subprocess(
                                binary_path=binary_path,
                                kernel_size_1=kernel_size_1,
                                kernel_size_2=kernel_size_2,
                                lab_processor=lab_processor,
                            )
                        ),
                        "kernel_size": [kernel_size_1, kernel_size_2],
                    }
                )

        for i in range(self.k_times):
            result = await tasks[i]["task"]
            tasks[i] = {**tasks[i], **result}
            tasks[i]["time_exe_ms_from_start_run_time_bin_name"] = (
                time.time() - tasks[i]["time_st"]
            ) * 1000
            print(
                f'[{tasks[i]["idx_run_time"]}] finished with `time_kernel_exe_ms`: {tasks[i]["time_kernel_exe_ms"]} ms'
            )

        # print stats
        await print_stat_time([item.get("time_kernel_exe_ms") for item in tasks])

        if all(item.get("test_verification_result") for item in tasks):
            df_scores = pd.DataFrame(
                [
                    {k: v for k, v in item.items() if k not in ("time_st", "task")}
                    for item in tasks
                ]
            )
            df_scores.to_csv(
                os.path.join(dir2save, f"stats_{bin_name}.csv"), index=False
            )
            print("SUCCESS!")
        else:
            df_failed = pd.DataFrame(
                [
                    {k: v for k, v in item.items() if k not in ("time_st", "task")}
                    for item in tasks
                    if not item.get("test_verification_result")
                ]
            )
            print(f"FAILED: len={df_failed.shape[0]}!")

            df_failed.to_csv(
                os.path.join(dir2save, f"failed_{bin_name}.csv"), index=False
            )
        return df_scores

    async def run_experiments(
        self, lab_processor: BaseLabProcessor
    ) -> Optional[pd.DataFrame]:
        st = time.time()
        run_experiment_cpu_task = None
        print(f"[Experiments] START")
        run_experiment_cuda_task = asyncio.create_task(
            self.run_experiment(
                binary_path=self.binary_path_cuda,
                kernel_sizes=self.kernel_sizes,
                lab_processor=lab_processor,
            )
        )

        if self.binary_path_cpu:
            # df_scores_cpu = await self.run_experiment(
            run_experiment_cpu_task = asyncio.create_task(
                self.run_experiment(
                    binary_path=self.binary_path_cpu,
                    kernel_sizes=[[None, None]],
                    lab_processor=lab_processor,
                )
            )

        df_scores = await run_experiment_cuda_task
        df_scores["device"] = ["CUDA" for _ in range(df_scores.shape[0])]
        if self.binary_path_cpu and run_experiment_cpu_task:
            df_scores_cpu = await run_experiment_cpu_task
            df_scores_cpu["device"] = ["CPU" for _ in range(df_scores_cpu.shape[0])]
            df_scores = pd.concat([df_scores, df_scores_cpu], ignore_index=True)

        print(f"[Experiments] FINISH time exe: {time.time() - st}")
        return df_scores
