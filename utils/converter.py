import binascii
import ctypes
import io
import os
import struct

from PIL import Image

from utils import get_size


class ImgData:
    def __init__(self, path2data: str, idx = None):
        self.idx = idx
        self.c_data_bytes = None
        self.c_data_bytes_path = None
        self.hex = None
        self.hex_path = None
        self.png = None
        self.png_path = None
        self.size = None

        assert os.path.exists(path2data), f"os.path.exists({path2data}) is False"
        self.dir2save = os.path.dirname(path2data)
        self.data_name = os.path.splitext(os.path.basename(path2data))[0]
        self.data_ext = os.path.splitext(os.path.basename(path2data))[1]

        if path2data.endswith(".data"):
            self.c_data_bytes: bytes = self._from_c_data_file(path2data)
            self.c_data_bytes_path: str = path2data
            self.hex = self._to_hex()
            self.hex_path = self._to_hex_file(self.hex)
            self.png = self._to_png()
            self.png_path = self._to_png_file(self.png)

        elif path2data.endswith(".png"):
            self.c_data_bytes: bytes = self._from_png_file(path2data)
            self.c_data_bytes_path: str = self._to_c_data_file()
            self.hex = self._to_hex()
            # self.png = self._to_png()
            self.hex_path = self._to_hex_file(self.hex)
            self.png_path = path2data

        elif path2data.endswith(".txt"):
            self.c_data_bytes: bytes = self._from_hex_file(path2data)
            self.c_data_bytes_path: str = self._to_c_data_file()
            # self.hex = self._to_hex()
            self.png = self._to_png()
            self.png_path = self._to_png_file(self.png)
            self.hex_path = path2data
        else:
            raise ValueError(
                f"`path2data` should endwith '.data' OR '.png' OR '.txt'. \nNOW: `path2data`={path2data}"
            )

        self.size: float = get_size(self.c_data_bytes) # KB

    def _from_c_data_file(self, path2data: str) -> bytes:
        with open(path2data, "rb") as f:
            c_data_bytes = f.read()
        return c_data_bytes

    def _to_c_data_file(
        self,
    ) -> str:
        c_data_bytes_path = os.path.join(self.dir2save, f"{self.data_name}.data")
        with open(c_data_bytes_path, "wb") as f:
            f.write(self.c_data_bytes)
        return c_data_bytes_path

    def _to_png(self) -> Image.Image:
        with io.BytesIO(self.c_data_bytes) as fin:
            (w, h) = struct.unpack("ii", fin.read(8))  # Read width and height
            buff = ctypes.create_string_buffer(4 * w * h)
            fin.readinto(buff)  # Read pixel data into buffer

        img = Image.new("RGBA", (w, h))
        pix = img.load()
        offset = 0
        for j in range(h):
            for i in range(w):
                (r, g, b, a) = struct.unpack_from("cccc", buff, offset)
                pix[i, j] = (ord(r), ord(g), ord(b), ord(a))
                offset += 4

        return img

    def _to_png_file(self, img: Image.Image) -> str:
        png_path = os.path.join(self.dir2save, f"{self.data_name}.png")
        img.save(png_path)
        return png_path

    def _from_png_file(self, path2data: str) -> bytes:
        img = Image.open(path2data)
        self.png = img
        (w, h) = img.size[0:2]
        pix = img.load()
        buff = ctypes.create_string_buffer(8 + 4 * w * h)
        offset = 0
        struct.pack_into("ii", buff, offset, w, h)
        offset = 8
        for j in range(h):
            for i in range(w):
                r = bytes((pix[i, j][0],))
                g = bytes((pix[i, j][1],))
                b = bytes((pix[i, j][2],))
                a = bytes((255,))
                struct.pack_into("cccc", buff, offset, r, g, b, a)
                offset += 4

        return buff.raw

    def _from_hex_file(self, path2data: str) -> bytes:
        # Read the hex data from the input file
        with open(path2data, "r") as file:
            hex_data = file.read()

        # Clean the hex data by removing spaces and newlines
        hex_data_cleaned = hex_data.replace("\n", "").replace(" ", "")
        self.hex = hex_data.replace("\n", " ")

        # Convert the hex data to binary format
        binary_data = binascii.unhexlify(hex_data_cleaned)

        return binary_data

    def _to_hex(self) -> str:
        # Convert binary data to hex representation
        hex_data = binascii.hexlify(self.c_data_bytes).decode("utf-8")

        # Format hex data with spaces for readability (optional)
        hex_data_formatted = " ".join(
            hex_data[i : i + 8] for i in range(0, len(hex_data), 8)
        )
        return hex_data_formatted

    def _to_hex_file(
        self,
        hex: str,
    ) -> str:
        hex_path = os.path.join(self.dir2save, f"{self.data_name}.txt")
        with open(hex_path, "w") as out_file:
            out_file.write(hex)
        return hex_path
