import asyncio
import os
from glob import glob
from typing import Optional, List
from uuid import uuid4

from tester import BaseLabProcessor
from utils import download_file, ImgData


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
        self.data2test_pre = (
            glob(os.path.join(dir_to_data, "*.png"))
            + glob(os.path.join(dir_to_data, "*.data"))
            + glob(os.path.join(dir_to_data, "*.txt"))
        ) + [
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
        self.current_index = 0
        self.lock = asyncio.Lock()

    async def get_next_item(self):
        async with self.lock:
            item = self.data_input[self.current_index]
            print(f"[DEBUG INFO] got index: self.data_input[{self.current_index}]. Is None: {bool(item is None)}")
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
            },
            {
                "filename": f"{next_item.data_name}.{next_item.data_ext}",
            },
        )

    async def verify_result(self, task_result: ImgData, **kwargs) -> bool:
        test_verification_result = True
        idx_data = kwargs.get("idx_data")
        task_result.idx = idx_data
        if idx_data in self.data_output_gt:
            item_output_gt: ImgData = self.data_output_gt[idx_data]
            hex_a = task_result.hex.replace("\n", "").replace(" ", "")
            hex_b = item_output_gt.hex.replace("\n", "").replace(" ", "")
            test_verification_result = bool(hex_a == hex_b)

        # TODO: Compare `item_res` and `item_output_gt` in another way
        return test_verification_result

    async def get_task_result(self, task_result_string: str) -> ImgData:
        return ImgData(path2data=task_result_string)

    def get_attr(self):
        return {
            "precision_array": self.precision_array,
            "atol": self.atol,
        }
