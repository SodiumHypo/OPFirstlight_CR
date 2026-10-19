"""Find all BL (branch-link) call sites targeting a vaddr in an ARM64 ELF.

Usage: python xref_bl.py <lib.so> <target_vaddr_hex> [max_results]
Pure word-scan (no capstone needed): BL = 0x94000000 | imm26.
"""
import struct
import sys


def load_exec_segment(path):
    with open(path, "rb") as f:
        data = f.read(0x2000000)
    e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum = struct.unpack_from("<H", data, 0x38)[0]
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_flags = struct.unpack_from("<II", data, off)
        p_offset, p_vaddr, _, p_filesz = struct.unpack_from("<QQQQ", data, off + 8)
        if p_type == 1 and (p_flags & 1):  # PT_LOAD executable
            return data, p_vaddr, p_offset, p_filesz
    return data, 0, 0, len(data)


def main():
    path, target = sys.argv[1], int(sys.argv[2], 16)
    max_results = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    data, seg_va, seg_off, seg_size = load_exec_segment(path)
    hits = []
    for i in range(0, seg_size - 4, 4):
        word = struct.unpack_from("<I", data, seg_off + i)[0]
        if (word >> 26) == 0x25:  # BL
            imm = word & 0x3FFFFFF
            if imm & 0x2000000:
                imm -= 0x4000000
            site = seg_va + i
            if site + imm * 4 == target:
                hits.append(site)
                if len(hits) >= max_results:
                    break
    for h in hits:
        print(f"bl from 0x{h:08x}")


if __name__ == "__main__":
    main()
