import argparse
import asyncio
import json
import os.path

from arg_parsing import hundle_unkown
from lab1.lab1_processor import Lab1Processor
from tester import BaseTester

MAP_LAB_PROCESSORS = {"lab1": Lab1Processor}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run kernel testing with subprocess.")
    parser.add_argument(
        "--binary_path_cuda", type=str, help="Path to the cuda binary file."
    )
    parser.add_argument(
        "--binary_path_cpu",
        type=str,
        help="Path to the cpu binary file.",
        default=None,
    )
    parser.add_argument(
        "--k_times", type=int, help="Number of times to run the kernel.", default=20
    )
    parser.add_argument(
        "--return_inp", help="Return input for binary.",action='store_true'
    )
    parser.add_argument(
        "--return_task_res", help="Return the result of task.", action='store_true'
    )
    parser.add_argument(
        "--kernel_sizes",
        type=str,
        default="[[512, 512]]",
        help="Optional kernel sizes in JSON format, e.g., '[[1, 32], [512, 512], [1024, 1024]]'",
    )

    args, unknown = parser.parse_known_args()
    kwargs = hundle_unkown(unknown)
    kernel_sizes = json.loads(args.kernel_sizes) if args.kernel_sizes else None

    lab_name: str = os.path.basename(
        os.path.dirname(os.path.dirname(args.binary_path_cuda))
    )

    print(f"Params:")
    print(f"return_inp=<{args.return_inp}>")
    print(f"return_task_res=<{args.return_task_res}>")
    print(f"lab_name=<{lab_name}>")
    print(f"binary_path_cuda=<{args.binary_path_cuda}>")
    print(f"binary_path_cpu=<{args.binary_path_cpu}>")
    print(f"k_times=<{args.k_times}>")
    print(f"kernel_sizes=<{kernel_sizes}>")
    print(f"kwargs=<{json.dumps(kwargs, indent=2)}>")

    tester = BaseTester(
        binary_path_cuda=args.binary_path_cuda,
        binary_path_cpu=args.binary_path_cpu,
        k_times=args.k_times,
        kernel_sizes=kernel_sizes,
        return_inp=args.return_inp,
        return_task_res=args.return_task_res
    )
    lab_processor = MAP_LAB_PROCESSORS[lab_name](**kwargs)
    asyncio.run(tester.run_experiments(lab_processor=lab_processor))
