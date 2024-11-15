import asyncio
import ctypes
import os
import struct
from typing import Optional

from PIL import Image


async def from_png_to_c_data(path2file: str, dir2save: Optional[str] = None) -> str:
    fn_splitted = os.path.splitext(os.path.basename(path2file))
    assert fn_splitted[1] == ".png", f"`path2file` be .png, {fn_splitted}"

    if dir2save is None:
        dir2save = os.path.dirname(path2file)

    path2save = os.path.join(dir2save, f"{fn_splitted[0]}.data")
    img = Image.open(path2file)
    (w, h) = img.size[0:2]
    pix = img.load()
    buff = ctypes.create_string_buffer(4 * w * h)
    offset = 0
    for j in range(h):
        for i in range(w):
            r = bytes((pix[i, j][0],))
            g = bytes((pix[i, j][1],))
            b = bytes((pix[i, j][2],))
            a = bytes((255,))
            struct.pack_into("cccc", buff, offset, r, g, b, a)
            offset += 4

    out = open(path2save, "wb")
    out.write(struct.pack("ii", w, h))
    out.write(buff.raw)
    out.close()
    return path2save


async def from_c_data_to_png(path2file: str, dir2save: Optional[str] = None) -> str:
    fn_splitted = os.path.splitext(os.path.basename(path2file))
    assert fn_splitted[1] == ".data", f"`path2file` should be .data, {fn_splitted}"

    if dir2save is None:
        dir2save = os.path.dirname(path2file)

    path2save = os.path.join(dir2save, f"{fn_splitted[0]}.png")

    fin = open(path2file, "rb")
    (w, h) = struct.unpack("ii", fin.read(8))
    buff = ctypes.create_string_buffer(4 * w * h)
    fin.readinto(buff)
    fin.close()
    img = Image.new("RGBA", (w, h))
    pix = img.load()
    offset = 0
    for j in range(h):
        for i in range(w):
            (r, g, b, a) = struct.unpack_from("cccc", buff, offset)
            pix[i, j] = (ord(r), ord(g), ord(b), ord(a))
            offset += 4

    img.save(path2save)
    return path2save


if __name__ == "__main__":
    async def main():
        path_png = "lab2/test_data/world_map.png"
        saved_path_data = await from_png_to_c_data(path_png)
        saved_path_png = await from_c_data_to_png(saved_path_data)
        print(f"saved_path_png={saved_path_png}")
        print(f"saved_path_data={saved_path_data}")

    asyncio.run(main())

