import numpy as np

from tester import BaseLabProcessor


class Lab1Processor(BaseLabProcessor):
    def __init__(
        self,
        seed: int = 42,
        max_vector_size: int = 3072,
        atol: float = 1e-10,
        precision_array: int = 10,
    ):
        super().__init__(seed=seed)
        self.double_left = -1e100
        self.double_right = 1e100
        self.max_vector_size = max_vector_size

        # pre proc param
        self.precision_array = precision_array

        # post proc param
        self.atol = atol

        self.vector_size = None
        self.first_vector = None
        self.second_vector = None

    async def pre_process(
        self,
    ):
        self.vector_size = np.random.randint(1, self.max_vector_size)
        self.first_vector = np.random.uniform(
            self.double_left, self.double_right, self.vector_size
        )
        self.second_vector = np.random.uniform(
            self.double_left, self.double_right, self.vector_size
        )
        first_vector_str = np.array2string(
            self.first_vector,
            separator=" ",
            max_line_width=np.inf,
            precision=self.precision_array,
        )[1:-1].strip()
        second_vector_str = np.array2string(
            self.second_vector,
            separator=" ",
            max_line_width=np.inf,
            precision=self.precision_array,
        )[1:-1].strip()
        return f"{self.vector_size}\n{first_vector_str}\n{second_vector_str}"

    async def verify_result(self, task_result: np.ndarray) -> bool:
        test_verification_result = np.allclose(
            task_result,
            self.first_vector - self.second_vector,
            atol=self.atol,
        )
        return test_verification_result

    async def get_task_result(self, task_result_string: str) -> np.ndarray:
        return np.fromstring(task_result_string, dtype=np.float64, sep=" ")

    async def get_data_state(self):
        return {
            "vector_size": self.vector_size,
            "first_vector": self.first_vector,
            "second_vector": self.second_vector,
        }
