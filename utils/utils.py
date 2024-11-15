# Get the size in bytes
import sys


def get_size(byte_data: bytes) -> float:
    size_in_bytes = sys.getsizeof(byte_data)
    return size_in_bytes / 1024  # KB
