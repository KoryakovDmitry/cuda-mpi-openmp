import asyncio
import os
import shutil

# from glob import glob
import random
from typing import Optional, List, Tuple
from uuid import uuid4

import numpy as np

from lab3.img_data_classifier import GroundTruthClass, MAX_CLASSES, get_random_pts
from tester import BaseLabProcessor
from utils import download_file, ImgData

NEW_LINE = "\n"
TOTAL_FILENAMES = [
    # lab2
    "stalker2.png",
    "98.data",
    "AoE.png",
    "doom.png",
    "hf2.png",
    "starcraft.png",
    "warcraft.png",
    "test_01.txt",
    "test_02.txt",
    "lenna.png",
    "57.data",
    "95.data",
    "99.data",
    "02.data",
    "96.data",
    "97.data",
    # lab3
    "04.data",
    "09.data",
    "test_01_lab3.txt",
    "test_02_lab3.txt",
]

MAP_TO_INIT_POINTS = {
    "test_01_lab3.txt": [
        GroundTruthClass(
            lbl=0, definition_points=np.array([[1, 2], [1, 0], [2, 2], [2, 1]])
        ),
        GroundTruthClass(
            lbl=1, definition_points=np.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        ),
    ],
}


class Lab3Processor(BaseLabProcessor):
    def __init__(
            self,
            seed: int = 42,
            atol: float = 1e-10,
            precision_array: int = 10,
            count_classes: int = None,
            count_pts: int = None,
            extra_links_to_png: Optional[List[str]] = [],
            dir_to_data: Optional[str] = "./lab3/data",
            dir_to_data_out: Optional[str] = None,
            dir_to_data_out_gt: Optional[str] = None,
    ):
        super().__init__(seed=seed)
        self.double_left = -1e100
        self.double_right = 1e100

        # pre proc param
        self.precision_array = precision_array

        # post proc param
        self.atol = atol

        self.data2test_pre = [
            os.path.join(dir_to_data, fn_i)
            for fn_i in TOTAL_FILENAMES
            if os.path.exists(os.path.join(dir_to_data, fn_i))
        ]

        if dir_to_data_out_gt is None:
            dir_to_data_out_gt = os.path.join(
                os.path.dirname(dir_to_data), f"{os.path.basename(dir_to_data)}_out_gt"
            )

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
            data_input_item = ImgData(path2data=path2data, idx=ii)
            (w, h) = data_input_item.png.size[0:2]

            # if os.path.basename(path2data) in MAP_TO_INIT_POINTS:
            #     definition_classes = MAP_TO_INIT_POINTS[os.path.basename(path2data)]
            # else:
            #     if count_classes is None:
            #         count_classes = random.randint(1, MAX_CLASSES + 1)
            #
            #     definition_classes = [
            #         GroundTruthClass(
            #             lbl=lbl, definition_points=get_random_pts(w=w, h=h, count_pts=count_pts)
            #         )
            #         for lbl in range(0, count_classes + 1)
            #     ]
            definition_classes = MAP_TO_INIT_POINTS["test_01_lab3.txt"]

            if not (
                    (len(definition_classes) <= MAX_CLASSES)
                    and (len(definition_classes) > 0)
            ):
                raise ValueError(
                    "SHOULD BE (len(definition_classes) <= MAX_CLASSES) and (len(definition_classes) > 0)"
                    f"NOW: ({len(definition_classes)} <= {MAX_CLASSES}) and ({len(definition_classes)} > 0)"
                )

            self.data_input[ii]: Tuple[ImgData, List[GroundTruthClass]] = (
                data_input_item,
                definition_classes,
            )

            get_path = lambda ext: os.path.join(
                dir_to_data_out_gt, f"{data_input_item.data_name}.{ext}"
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

        if dir_to_data_out is None:
            dir_to_data_out = os.path.join(
                os.path.dirname(dir_to_data), f"{os.path.basename(dir_to_data)}_out"
            )

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

    async def pre_process(self, **kwargs):
        device_info = kwargs.get(f"device_info")
        dir_to_data_out_with_device = os.path.join(self.dir_to_data_out, device_info)
        if not os.path.exists(dir_to_data_out_with_device):
            os.makedirs(dir_to_data_out_with_device, exist_ok=True)

        next_item, definition_classes = await self.get_next_item()
        out_path_res = os.path.join(
            dir_to_data_out_with_device, f"{next_item.data_name}.data"
        )

        pts_row = f"\n".join([
            f"{definition_class.definition_points.shape[0]} {np.array2string(definition_class.definition_points.reshape(-1), separator=' ', max_line_width=np.inf, precision=self.precision_array, )[1:-1].strip()}"
            for definition_class in definition_classes])
        return (
            f"{next_item.c_data_bytes_path}\n{out_path_res}\n{len(definition_classes)}\n{pts_row}",
            {
                "idx_data": next_item.idx,
                "out_path_res": out_path_res,
            },
            {
                "filename": f"{next_item.data_name}{next_item.data_ext} ({next_item.size:.5f} KB)",
                "stat_init_pts": f"Count Classes: {len(definition_classes)}, Count Points: {set((d.definition_points.shape[0] for d in definition_classes))}",
            },
        )

    async def verify_result(self, task_result: ImgData, **kwargs) -> bool:
        test_verification_result = True
        idx_data = kwargs.get("idx_data")
        task_result.idx = idx_data
        if idx_data in self.data_output_gt:
            item_output_gt: ImgData = self.data_output_gt[idx_data]
            data_input_item_full: Tuple[ImgData, List[GroundTruthClass]] = self.data_input[idx_data]
            data_input_item, definition_classes = data_input_item_full
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
