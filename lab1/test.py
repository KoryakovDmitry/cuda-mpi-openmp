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
    binary_path: str, n: int, first_vector: np.ndarray, second_vector: np.ndarray
):
    first_vector_str = np.array2string(
        first_vector, separator=" ", max_line_width=np.inf, precision=10
    )[1:-1].strip()
    second_vector_str = np.array2string(
        second_vector, separator=" ", max_line_width=np.inf, precision=10
    )[1:-1].strip()
    try:
        result = subprocess.run(
            [binary_path],
            input=f"{n}\n{first_vector_str}\n{second_vector_str}",
            text=True,
            capture_output=True,
            check=True,
        )
        result_vector = np.fromstring(result.stdout.strip(), dtype=np.float64, sep=" ")
        test_result = np.allclose(
            result_vector,
            first_vector - second_vector,
            atol=atol,
        )
        return True, result_vector, test_result, None
    except subprocess.CalledProcessError as e:
        err = e.stderr.strip()
        return False, None, None, err


async def run_binary(binary_path: str, k_times: int, max_vector_size: int):
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
                "time_exe": None,
                "task": asyncio.create_task(
                    run_subprocess(
                        binary_path=binary_path,
                        n=n,
                        first_vector=first_vector,
                        second_vector=second_vector,
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
        tasks[i]["time_exe"] = (time.time() - tasks[i]["time_st"]) * 1000
        print(f'[{tasks[i]["idx"]}] finished with : {tasks[i]["time_exe"]} ms')

    await get_stat_time([item.get("time_exe") for item in tasks])

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
                        "time_exe": fail.get("time_exe"),
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
    if len(sys.argv) != 4:
        print("Usage: python script.py <binary_path> <k_times> <max_vector_size>")
        # sys.exit(1)
        binary_path = "lab1/src/a.out"
        k_times = 10
        # max_vector_size = 2**25
        max_vector_size = 100
    else:
        binary_path = sys.argv[1]
        k_times = int(sys.argv[2])
        max_vector_size = int(sys.argv[3])

    asyncio.run(run_binary(binary_path, k_times, max_vector_size))
