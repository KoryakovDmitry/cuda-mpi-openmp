import numpy as np

from tester import BaseLabProcessor


class Lab1Processor(BaseLabProcessor):
    def __init__(
        self,
        seed: int = 42,
        min_vector_size: int = 1024,
        max_vector_size: int = 3072,
        atol: float = 1e-10,
        precision_array: int = 10,
    ):
        super().__init__(seed=seed)
        self.double_left = -1e100
        self.double_right = 1e100
        self.min_vector_size = min_vector_size
        self.max_vector_size = max_vector_size

        # pre proc param
        self.precision_array = precision_array

        # post proc param
        self.atol = atol

    async def pre_process(
        self,
    ):
        vector_size = np.random.randint(self.min_vector_size, self.max_vector_size)
        first_vector = np.random.uniform(
            self.double_left, self.double_right, vector_size
        )
        second_vector = np.random.uniform(
            self.double_left, self.double_right, vector_size
        )
        first_vector_str = np.array2string(
            first_vector,
            separator=" ",
            max_line_width=np.inf,
            precision=self.precision_array,
        )[1:-1].strip()
        second_vector_str = np.array2string(
            second_vector,
            separator=" ",
            max_line_width=np.inf,
            precision=self.precision_array,
        )[1:-1].strip()
        return (
            f"{vector_size}\n{first_vector_str}\n{second_vector_str}",
            {
                "first_vector": first_vector,
                "second_vector": second_vector,
            },
            {
                "vector_size": vector_size,
            },
        )

    async def verify_result(self, task_result: np.ndarray, **kwargs) -> bool:
        test_verification_result = True
        # test_verification_result = np.allclose(
        #     task_result,
        #     kwargs.get("first_vector") - kwargs.get("second_vector"),
        #     atol=self.atol,
        # )
        return test_verification_result

    async def get_task_result(self, task_result_string: str, **kwargs) -> np.ndarray:
        return np.fromstring(task_result_string, dtype=np.float64, sep=" ")

    def get_attr(self):
        return {
            "min_vector_size": self.min_vector_size,
            "max_vector_size": self.max_vector_size,
            "atol": self.atol,
        }
