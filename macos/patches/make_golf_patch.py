"""Generate a shadPS4 patch moving Everybody's Golf's hard-coded memory regions out of the
Apple Silicon GPU carve-out (0x1000000000-0x6FFFFFFFFF), which shadPS4 cannot map under Rosetta.

The game's allocator classifies every pointer by (addr & 0xFFFFFFF000000000) against the region
bases 0x1000000000, 0x2000000000 and 0x3000000000, and gets a null heap for anything else. Every
`movabs reg, <base>` in the code is rewritten so the bases become 0x8000000000, 0x9000000000 and
0xA000000000, which are inside shadPS4's macOS user range.

usage: make_golf_patch.py <eboot.bin> <app_ver> <out.xml>
"""
import re
import struct
import sys

REMAP = {0x10: 0x80, 0x20: 0x90, 0x30: 0xA0}  # byte 4 of the 64-bit immediate
BASE = 0x400000
# Immediates that match the pattern but are not region bases (addresses are of the patched byte).
# Of 01.17's 444 matches, 437 are bases: 431 sit in the allocator's three-way compare chains and six
# are heap bases passed to its region setup. The others:
#   0x14e25cc `movabs rax, 0x1000000000` initialises the sound output table's first entry to
#             {id 0, format 0x10}; at 0x80 the mixer rejects the format and the game is silent.
#   0xaa1ec1  stores {0, 16} into a game object's field at +0x10 (its owner holds the same 16).
#   0x138ce2c, 0x13be52b  are the second qword of a {addr, len, mtype, prot} block that callback
#             0x138d5c0 hands to a memory-protect call: the region's 64 GiB *length*, which equals
#             region 1's base. The first qword is the region address and stays patched.
EXCLUDE = {"01.17": {0x14E25D2, 0xAA1EC7, 0x138CE32, 0x13BE531}}

path, app_ver, out = sys.argv[1:4]
d = open(path, "rb").read()
count, = struct.unpack_from("<H", d, 0x18)
segs = [struct.unpack_from("<QQQQ", d, 0x20 + i * 0x20) for i in range(count)]
elf = 0x20 + count * 0x20
phoff, = struct.unpack_from("<Q", d, elf + 0x20)
phnum, = struct.unpack_from("<H", d, elf + 0x38)
ph = [struct.unpack_from("<IIQQQQQQ", d, elf + phoff + i * 56) for i in range(phnum)]

text = None
for flags, off, size, _ in segs:
    idx = (flags >> 20) & 0xFFF
    if flags & 0x800 and ph[idx][0] == 1 and ph[idx][1] & 1:  # loadable, executable
        text = (d[off:off + size], BASE + ph[idx][3])
        break
code, vbase = text

lines = []
for hi, new in REMAP.items():
    imm = (hi << 32).to_bytes(8, "little")
    for m in re.finditer(re.escape(imm), code):
        s = m.start() - 2
        if s >= 0 and code[s] in (0x48, 0x49) and 0xB8 <= code[s + 1] <= 0xBF:
            addr = vbase + s + 6  # REX, opcode, then imm bytes 0..7; byte 4 is the base
            if addr not in EXCLUDE.get(app_ver, set()):
                lines.append((addr, new))
lines.sort()

xml = ['<?xml version="1.0" encoding="utf-8"?>', "<Patch>"]
xml.append(
    f'    <Metadata Title="Everybody\'s Golf" Name="macOS Memory Workaround" '
    f'Note="Moves the allocator\'s region bases 0x10/0x20/0x30 GiB to 0x80/0x90/0xA0 GiB, out of the '
    f'Apple Silicon GPU carve-out that shadPS4 cannot map under Rosetta. {len(lines)} immediates." '
    f'Author="kirbycope" PatchVer="1.0" AppVer="{app_ver}" AppElf="eboot.bin" isEnabled="true">'
)
xml.append("        <PatchList>")
for addr, val in lines:
    xml.append(f'            <Line Type="bytes" Address="{addr:#x}" Value="{val:02X}"/>')
xml.append("        </PatchList>")
xml.append("    </Metadata>")
xml.append("</Patch>")
open(out, "w").write("\n".join(xml) + "\n")
print(f"{len(lines)} lines for {app_ver} -> {out}")
