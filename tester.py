import asyncio
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    debug_data: Optional[Dict[str, Any]]


class BaseLabProcessor:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)

    def get_attr(
        self,
    ) -> Dict[str, Any]:
        pass

    async def pre_process(self, **kwargs) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        pass

    async def verify_result(self, task_result: Any, **kwargs) -> bool:
        pass

    async def get_task_result(self, task_result_string: str) -> Any:
        pass

    async def post_process(self, result_stdout: str, **kwargs) -> TaskResult:
        result_stdout_split = result_stdout.split("\n")
        time_kernel_exe_ms_task = asyncio.create_task(
            get_time_kernel_exe(result_stdout_split[0])
        )

        task_result_string = "\n".join(result_stdout_split[1:])
        task_result = await self.get_task_result(task_result_string)
        test_verification_result = await self.verify_result(task_result, **kwargs)
        return TaskResult(
            time_kernel_exe_ms=await time_kernel_exe_ms_task,
            test_verification_result=test_verification_result,
            task_result=task_result,
        )


async def run_subprocess(
    binary_path: str,
    lab_processor: BaseLabProcessor,
    return_inp: bool,
    kernel_size_1: Optional[int] = None,
    kernel_size_2: Optional[int] = None,
) -> SubProcessResult:
    try:
        input_str, inter_data_to_verify, debug_data = await lab_processor.pre_process()
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
            result_stdout=result.stdout, **inter_data_to_verify
        )
        if return_inp:
            debug_data["input_str"] = input_str
        return SubProcessResult(
            test_verification_result=task_result.test_verification_result,
            task_result=task_result.task_result,
            time_kernel_exe_ms=task_result.time_kernel_exe_ms,
            debug_data=debug_data,
            status=True,
            err=None,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr.strip()
        return SubProcessResult(
            test_verification_result=None,
            task_result=None,
            time_kernel_exe_ms=None,
            debug_data=None,
            status=False,
            err=err,
        )


class BaseTester:
    def __init__(
        self,
        binary_path_cuda: str,
        k_times: int,
        kernel_sizes: List[List[Optional[int]]],
        metadata_columns2plot: List[str],
        binary_path_cpu: Optional[str] = None,
        return_inp: bool = False,
        return_task_res: bool = False,
    ):
        self.binary_path_cuda = binary_path_cuda
        self.binary_path_cpu = binary_path_cpu
        self.metadata_columns2plot = metadata_columns2plot
        self.dir2save = os.path.dirname(binary_path_cuda)
        self.kernel_sizes = kernel_sizes
        self.k_times = k_times
        self.return_inp = return_inp
        self.return_task_res = return_task_res

    async def run_experiment(
        self,
        binary_path: str,
        kernel_sizes: List[List[Optional[int]]],
        lab_processor: BaseLabProcessor,
    ) -> pd.DataFrame:
        bin_name = os.path.splitext(os.path.basename(binary_path))[0]

        print(f"[Experiment bin_name=<{bin_name}>] START")
        df_scores = pd.DataFrame()
        tasks = []

        for i in range(self.k_times):
            for kernel_size_1, kernel_size_2 in kernel_sizes:
                print(
                    f"[Experiment bin_name=<{bin_name}> task={i} kernel_size=<{[kernel_size_1, kernel_size_2]}>] started"
                )
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
                                return_inp=self.return_inp,
                            )
                        ),
                        "kernel_size": [kernel_size_1, kernel_size_2],
                    }
                )

        for task_i in range(len(tasks)):
            result: SubProcessResult = await tasks[task_i]["task"]
            result_dict = asdict(result)
            tasks[task_i] = {
                **tasks[task_i],
                **{k: v for k, v in result_dict.items() if k != "debug_data"},
                **lab_processor.get_attr(),
                **result.debug_data,
            }
            tasks[task_i]["time_exe_ms_from_start_run_time_bin_name"] = (
                time.time() - tasks[task_i]["time_st"]
            ) * 1000
            print(
                f'[Experiment bin_name=<{bin_name}> task={tasks[task_i]["idx_run_time"]} kernel_size=<{tasks[task_i]["kernel_size"]}>] finished with `time_kernel_exe_ms`: {tasks[task_i]["time_kernel_exe_ms"]} ms'
            )

        if all(item.get("test_verification_result") for item in tasks):
            # print stats
            await print_stat_time([item.get("time_kernel_exe_ms") for item in tasks])
            filtter = ("time_st", "task") if self.return_task_res else ("time_st", "task", "task_result")
            df_scores = pd.DataFrame(
                [
                    {k: v for k, v in item.items() if k not in filtter}
                    for item in tasks
                ]
            )
            df_scores.to_csv(
                os.path.join(self.dir2save, f"stats_{bin_name}.csv"), index=False
            )
            print(f"[Experiment bin_name=<{bin_name}>] SUCCESS!")
        else:
            df_failed = pd.DataFrame(
                [
                    {k: v for k, v in item.items() if k not in ("time_st", "task")}
                    for item in tasks
                    if not item.get("test_verification_result")
                ]
            )
            print(
                f"[Experiment bin_name=<{bin_name}>] FAILED: len={df_failed.shape[0]}!"
            )

            df_failed.to_csv(
                os.path.join(self.dir2save, f"failed_{bin_name}.csv"), index=False
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

        if df_scores.shape[0] > 0:
            self.plot_df_score(df_scores=df_scores)

        print(f"[Experiments] FINISH time exe: {time.time() - st}")
        return df_scores

    def plot_df_score(self, df_scores: pd.DataFrame, ):
        # 1. Group by 'device' and 'kernel_size', calculate median 'time_kernel_exe_ms'
        df_scores['kernel_size'] = df_scores['kernel_size'].apply(lambda x: json.dumps(x))
        grouped = df_scores.groupby(['device', 'kernel_size']).agg(
            median_time=('time_kernel_exe_ms', 'median'),
            sample_count=('time_kernel_exe_ms', 'size')
        ).reset_index()

        # 2. Format the labels
        def format_label(row):
            if row['device'] == 'CPU':
                return f"{row['device']}"
            else:
                return f"{row['device']}_{row['kernel_size']}"

        grouped['label'] = grouped.apply(format_label, axis=1)

        # 3. Collect unique metadata values for the legend
        legend_text = ""
        for col in self.metadata_columns2plot:
            unique_values = df_scores[col].unique()
            unique_values_str = ', '.join(map(str, unique_values))
            legend_text += f"{col}: [{unique_values_str}]\n"

        # Add sample count information to the legend
        legend_text += "\nSample Count by Group:\n"
        for _, row in grouped.iterrows():
            legend_text += f"{row['label']}: {row['sample_count']} samples\n"

        # 4. Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the bar plot
        bars = ax.bar(grouped['label'], grouped['time_kernel_exe_ms'], color='skyblue')

        # Add median values on top of each bar with a smaller offset
        for bar, median in zip(bars, grouped['time_kernel_exe_ms']):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{median:.5f}", ha='center', va='bottom')

        # Place the legend text outside the plot area
        plt.text(1.02, 0.95, legend_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        # Set labels and title
        ax.set_xlabel('Device and Kernel Size')
        ax.set_ylabel('Median Execution Time (ms)')
        ax.set_title('Median Execution Time by Device and Kernel Size')

        # Adjust layout to make room for the legend
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(os.path.join(self.dir2save, f'median_execution_time.png'), dpi=300, bbox_inches='tight')

        # If you want to close the plot to free up memory
        plt.close()
