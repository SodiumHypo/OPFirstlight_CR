"""Disassemble a range of an ARM64 ELF .so at a virtual offset.

Usage: python disasm_arm64.py <lib.so> <vaddr_hex> [count]
Annotates ADRP+LDR pairs to resolve static-global accesses (for root-slot
discovery on libg.so builds). Read-only tool; part of pipeline P0.
"""
import struct
import sys


def load_segments(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"\x7fELF", "not an ELF"
    e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum = struct.unpack_from("<H", data, 0x38)[0]
    segs = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_flags = struct.unpack_from("<II", data, off)
        p_offset, p_vaddr, p_paddr, p_filesz = struct.unpack_from("<QQQQ", data, off + 8)
        if p_type == 1:  # PT_LOAD
            segs.append((p_vaddr, p_vaddr + p_filesz, p_offset))
    return data, segs


def vaddr_to_offset(segs, vaddr):
    for lo, hi, off in segs:
        if lo <= vaddr < hi:
            return off + (vaddr - lo)
    return None


def main():
    path, vaddr = sys.argv[1], int(sys.argv[2], 16)
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN

    data, segs = load_segments(path)
    off = vaddr_to_offset(segs, vaddr)
    if off is None:
        print(f"vaddr 0x{vaddr:x} not in any PT_LOAD")
        return
    code = data[off : off + count * 4]
    md = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    md.detail = False
    adr_pages = {}
    for insn in md.disasm(code, vaddr):
        note = ""
        if insn.mnemonic == "adrp":
            try:
                reg, imm = insn.op_str.split(", ")
                adr_pages[reg] = int(imm, 0)
            except ValueError:
                pass
        elif insn.mnemonic == "ldr" and "#" in insn.op_str:
            base = insn.op_str.split("[")[1].rstrip("]").split(",")[0].strip()
            if base in adr_pages:
                try:
                    disp = int(insn.op_str.split("#")[1].rstrip("]"), 0)
                    note = f"  ; global @0x{adr_pages[base] + disp:x}"
                except ValueError:
                    pass
        print(f"0x{insn.address:08x}: {insn.bytes.hex():<12} {insn.mnemonic:<8} {insn.op_str}{note}")


if __name__ == "__main__":
    main()
