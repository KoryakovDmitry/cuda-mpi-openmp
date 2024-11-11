import asyncio
import json
import os
import shutil
import subprocess
import time

import numpy as np
import sys
from typing import Tuple, List

np.random.seed(42)

# double_left = -1.7976931348623157e308
# double_right = 1.7976931348623157e308
double_left = -1e100
double_right = 1e100
atol = 1e-10
pattern = r"execution time: <([\d.]+) ms>"


async def get_time_exe(text: str, pattern: str = pattern) -> float | None:
    # Search for the pattern
    match = re.search(pattern, text)

    # Extract and convert to float
    if match:
        time_ms = float(match.group(1))
        return time_ms
    return None


async def get_stat_time(data_np):
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


async def run_subprocess(
    binary_path: str,
    n: int,
    first_vector: np.ndarray,
    second_vector: np.ndarray,
    kernel_size_1: int = None,
    kernel_size_2: int = None,
):
    first_vector_str = np.array2string(
        first_vector, separator=" ", max_line_width=np.inf, precision=10
    )[1:-1].strip()
    second_vector_str = np.array2string(
        second_vector, separator=" ", max_line_width=np.inf, precision=10
    )[1:-1].strip()
    try:
        if kernel_size_1 and kernel_size_2:
            input_str = f"{kernel_size_1}\n{kernel_size_2}\n{n}\n{first_vector_str}\n{second_vector_str}"
        else:
            input_str = f"{n}\n{first_vector_str}\n{second_vector_str}"

        result = subprocess.run(
            [binary_path],
            input=input_str,
            text=True,
            capture_output=True,
            check=True,
        )
        result_split = result.stdout
        result_vector = np.fromstring(
            result_split[1].strip(), dtype=np.float64, sep=" "
        )
        test_result = np.allclose(
            result_vector,
            first_vector - second_vector,
            atol=atol,
        )
        time_exe_ms_calc = await get_time_exe(result_split[0])
        return True, result_vector, test_result, None, time_exe_ms_calc
    except subprocess.CalledProcessError as e:
        err = e.stderr.strip()
        return False, None, None, err, None


async def run_binary(
    binary_path: str,
    k_times: int,
    max_vector_size: int,
    kernel_size_1: int = None,
    kernel_size_2: int = None,
):
    tasks = []
    for i in range(k_times):
        n = np.random.randint(1, max_vector_size)

        first_vector = np.random.uniform(double_left, double_right, n)

        second_vector = np.random.uniform(double_left, double_right, n)

        print(f"[{i}] started")
        tasks.append(
            {
                "idx": i,
                "time_st": time.time(),
                "time_exe_ms_full": None,
                "time_exe_ms_calc": None,
                "task": asyncio.create_task(
                    run_subprocess(
                        binary_path=binary_path,
                        n=n,
                        first_vector=first_vector,
                        second_vector=second_vector,
                        kernel_size_1=kernel_size_1,
                        kernel_size_2=kernel_size_2,
                    )
                ),
                "n": n,
                "first_vector": first_vector,
                "second_vector": second_vector,
                "status": None,
                "result_vector": None,
                "test_result": None,
                "err": None,
            }
        )

    for i in range(k_times):
        result = await tasks[i]["task"]
        tasks[i]["status"] = result[0]
        tasks[i]["result_vector"] = result[1]
        tasks[i]["test_result"] = result[2]
        tasks[i]["err"] = result[3]
        tasks[i]["time_exe_ms_full"] = (time.time() - tasks[i]["time_st"]) * 1000
        tasks[i]["time_exe_ms_calc"] = result[4]
        print(
            f'[{tasks[i]["idx"]}] finished with `time_exe_ms_calc`: {tasks[i]["time_exe_ms_calc"]} ms'
        )

    await get_stat_time([item.get("time_exe_ms_calc") for item in tasks])

    if all(item.get("test_result") for item in tasks):
        print("SUCCESS")
    else:
        failed = [item for item in tasks if not item.get("test_result")]
        print(f"FAILED: len={len(failed)}")

        dir2save = "./failed"
        os.makedirs(dir2save, exist_ok=True)
        shutil.rmtree(dir2save)
        os.makedirs(dir2save, exist_ok=True)
        for fail in failed:
            with open(
                os.path.join(dir2save, f'{fail["idx"]}_first_vector.npy'), "wb"
            ) as f:
                np.save(f, fail["first_vector"])

            with open(
                os.path.join(dir2save, f'{fail["idx"]}_second_vector.npy'), "wb"
            ) as f:
                np.save(f, fail["second_vector"])

            with open(
                os.path.join(dir2save, f'{fail["idx"]}_result_vector.npy'), "wb"
            ) as f:
                np.save(f, fail["result_vector"])

        with open("failed.json", "w", encoding="utf8") as f:
            json.dump(
                [
                    {
                        "idx": fail.get("idx"),
                        "time_exe_ms_full": fail.get("time_exe_ms_full"),
                        "time_exe_ms_calc": fail.get("time_exe_ms_calc"),
                        "n": fail.get("n"),
                        # "first_vector": fail.get("first_vector"),
                        # "second_vector": fail.get("second_vector"),
                        "status": fail.get("status"),
                        # "result_vector": fail.get("result_vector"),
                        "test_result": fail.get("test_result"),
                        "err": fail.get("err"),
                    }
                    for fail in failed
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    kernel_size_1 = None
    kernel_size_2 = None
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py <binary_path> <k_times> <max_vector_size> [<kernel_size_1>] [<kernel_size_2>]"
        )
        # sys.exit(1)
        binary_path = "lab1/src/a.out"
        k_times = 10
        # max_vector_size = 2**25
        max_vector_size = 100
    else:
        binary_path = sys.argv[1]
        k_times = int(sys.argv[2])
        max_vector_size = int(sys.argv[3])
        if os.path.splitext(binary_path)[1] == ".cu":
            kernel_size_1 = int(sys.argv[4])
            kernel_size_2 = int(sys.argv[5])

    asyncio.run(
        run_binary(binary_path, k_times, max_vector_size, kernel_size_1, kernel_size_2)
    )
