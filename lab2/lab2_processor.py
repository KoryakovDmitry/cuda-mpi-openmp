import asyncio
import os
import shutil

# from glob import glob
from typing import Optional, List
from uuid import uuid4

from tester import BaseLabProcessor
from utils import download_file, ImgData

NEW_LINE = "\n"


class Lab2Processor(BaseLabProcessor):
    def __init__(
        self,
        seed: int = 42,
        atol: float = 1e-10,
        precision_array: int = 10,
        extra_links_to_png: Optional[List[str]] = [],
        dir_to_data: Optional[str] = "./lab2/data/",
        dir_to_data_out: Optional[str] = "./lab2/data_out/",
        dir_to_data_out_gt: Optional[str] = "./lab2/data_out_gt/",
    ):
        super().__init__(seed=seed)
        self.double_left = -1e100
        self.double_right = 1e100

        # pre proc param
        self.precision_array = precision_array

        # post proc param
        self.atol = atol

        filenames = [
            "98.data",
            "test_01.txt",
            "test_02.txt",
            "lenna.png",
            "57.data",
            "95.data",
            "99.data",
            "02.data",
            "96.data",
            "97.data",
        ]
        self.data2test_pre = [os.path.join(dir_to_data, fn_i) for fn_i in filenames]

        # self.data2test_pre = (
        #     glob(os.path.join(dir_to_data, "*.png"))
        #     + glob(os.path.join(dir_to_data, "*.data"))
        #     + glob(os.path.join(dir_to_data, "*.txt"))
        # )
        self.data2test_pre += [
            download_file(
                extra_link_to_png, save_dir=dir_to_data, filename=f"{str(uuid4())}.png"
            )
            for extra_link_to_png in extra_links_to_png
        ]

        self.data_input = dict()
        self.data_output_gt = dict()
        for ii, path2data in enumerate(self.data2test_pre):
            self.data_input[ii] = ImgData(path2data=path2data, idx=ii)

            get_path = lambda ext: os.path.join(
                dir_to_data_out_gt, f"{self.data_input[ii].data_name}.{ext}"
            )

            path2data_gt = None
            if os.path.exists(get_path("txt")):
                path2data_gt = get_path("txt")

            elif os.path.exists(get_path("data")):
                path2data_gt = get_path("data")

            elif os.path.exists(get_path("png")):
                path2data_gt = get_path("png")

            if path2data_gt:
                self.data_output_gt[ii] = ImgData(path2data=path2data_gt, idx=ii)

        self.dir_to_data_out = dir_to_data_out
        os.makedirs(self.dir_to_data_out, exist_ok=True)
        shutil.rmtree(self.dir_to_data_out)
        os.makedirs(self.dir_to_data_out, exist_ok=True)
        self.current_index = 0
        self.lock = asyncio.Lock()

    async def get_next_item(self):
        async with self.lock:
            # print(f"[DEBUG INFO] got index: self.data_input[{self.current_index}]. Is self.current_index None: {bool(self.current_index is None)}")
            item = self.data_input[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.data_input)
            return item

    async def pre_process(
        self,
    ):
        next_item = await self.get_next_item()
        out_path_res = os.path.join(self.dir_to_data_out, f"{next_item.data_name}.data")
        return (
            f"{next_item.c_data_bytes_path}\n{out_path_res}",
            {
                "idx_data": next_item.idx,
                "out_path_res": out_path_res,
            },
            {
                "filename": f"{next_item.data_name}{next_item.data_ext}",
            },
        )

    async def verify_result(self, task_result: ImgData, **kwargs) -> bool:
        test_verification_result = True
        idx_data = kwargs.get("idx_data")
        task_result.idx = idx_data
        if idx_data in self.data_output_gt:
            item_output_gt: ImgData = self.data_output_gt[idx_data]
            data_input_item: ImgData = self.data_input[idx_data]
            hex_a = task_result.hex.replace("\n", "").replace(" ", "").upper()
            hex_b = item_output_gt.hex.replace("\n", "").replace(" ", "").upper()
            test_verification_result = bool(hex_a == hex_b)
            if not test_verification_result:
                print(
                    f"[verify_result] FAILED `verify_result`: `{task_result.data_name}`!"
                )
                print(
                    f"[verify_result] [input_data.hex] {data_input_item.hex.replace(NEW_LINE, ' ').upper()}"
                )
                print(
                    f"[verify_result] [task_result.hex] {task_result.hex.replace(NEW_LINE, ' ').upper()}"
                )
                print(
                    f"[verify_result] [ground_truth.hex] {item_output_gt.hex.replace(NEW_LINE, ' ').upper()}"
                )
                print(
                    f"[verify_result] SHOULD BE `task_result.hex` == `ground_truth.hex`!!!"
                )

        # TODO: Compare `item_res` and `item_output_gt` in another way
        return test_verification_result

    async def get_task_result(self, task_result_string: str, **kwargs) -> ImgData:
        return ImgData(path2data=kwargs.get("out_path_res"))

    def get_attr(self):
        return {
            "precision_array": self.precision_array,
            "atol": self.atol,
        }
