"""
lake_exporter.py — Write Lake Controller EQ Overlay (.ovl) files.

Reverse-engineered from Adamson library overlays (Lake Controller 6.5.0).
File layout:
  4-byte magic + directory of (32-byte name, 4-byte size, 4-byte offset) entries
  Fixed blocks: IOFileRev, ControllerAppRev, MinimumSupportedControllerRev,
                MinimumSupportedControllerName, SubSystem, ShowKey0,
                SingleOverlay (488 bytes), SingleSegments (80 bytes × n_bands),
                IOFileRevEndRev5_7, LakeEndOfFile

SingleSegments band block (80 bytes):
  0-27   zeros
  28-31  block index (uint32)
  32-47  4 × 0xFFFFFFFF (disabled-slot markers)
  48-49  band index (uint16, sequential from 0)
  50-51  filter type (uint16: 1=Peak, 2=HiShelf, 3=LoShelf)
  52-55  frequency L (float32, kHz)
  56-59  frequency R (float32, kHz, same as L)
  60-63  gain L (float32, dB)
  64-67  gain R (float32, dB, same as L)
  68-71  bandwidth (float32, octaves)
  72-79  zeros
"""

import struct
import math
import os


# ── Constants ────────────────────────────────────────────────────────────────

_FILE_MAGIC     = bytes([0xDD, 0xF0, 0x0A, 0x00])
_IO_FILE_REV    = struct.pack('<f', 10.5)
_CTRL_APP_REV   = struct.pack('<f', 65.007)          # Lake Controller 6.5
_MIN_CTRL_REV   = struct.pack('<f', 65.006)
_MIN_CTRL_NAME  = b'Lake Controller 6.5.0  \x00'     # 24 bytes
_SUBSYSTEM      = struct.pack('<I', 0)
_SHOW_KEY0      = b'ShowKey0'                         # 8 bytes
_IO_REV_END     = _IO_FILE_REV
_EOF_MARKER     = struct.pack('<I', 0)

_FILTER_TYPES = {
    'Peak':     1,
    'Hi Shelf': 2,
    'Lo Shelf': 3,
}


# ── Q ↔ Bandwidth conversion ─────────────────────────────────────────────────

def _q_to_bw(q: float) -> float:
    """Convert Q to bandwidth in octaves."""
    if q <= 0:
        return 1.0
    return 2.0 * math.asinh(1.0 / (2.0 * q)) / math.log(2.0)


# ── Block builders ───────────────────────────────────────────────────────────

def _build_band_block(block_idx: int, band: dict) -> bytes:
    """Build one 80-byte band block from an EQ band dict {freq, gain, q, type}."""
    ftype_code = _FILTER_TYPES.get(band.get('type', 'Peak'), 1)
    freq_khz   = float(band['freq']) / 1000.0
    gain_db    = float(band['gain'])
    bw         = _q_to_bw(float(band.get('q', 1.0)))

    block = bytearray(80)
    struct.pack_into('<I',  block, 28, block_idx)            # block index
    struct.pack_into('<4I', block, 32, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
    struct.pack_into('<H',  block, 48, block_idx)            # band index
    struct.pack_into('<H',  block, 50, ftype_code)           # filter type
    struct.pack_into('<f',  block, 52, freq_khz)             # freq L
    struct.pack_into('<f',  block, 56, freq_khz)             # freq R
    struct.pack_into('<f',  block, 60, gain_db)              # gain L
    struct.pack_into('<f',  block, 64, gain_db)              # gain R
    struct.pack_into('<f',  block, 68, bw)                   # bandwidth (oct)
    return bytes(block)


def _build_single_overlay(name: str) -> bytes:
    """Build the fixed 488-byte SingleOverlay metadata block."""
    block = bytearray(488)

    # Bytes 28-45: 18 × 0xFF (empty key)
    for i in range(28, 46):
        block[i] = 0xFF

    # Bytes 48-111: overlay name, null-padded to 64 bytes
    name_bytes = name.encode('ascii', errors='replace')[:63]
    block[48:48 + len(name_bytes)] = name_bytes

    # Bytes 176-179: flags (01 00 01 00)
    struct.pack_into('<HH', block, 176, 1, 1)

    # Bytes 204-207: flags (00 00 01 00)
    struct.pack_into('<I',  block, 204, 0x00010000)

    # Bytes 208-211: flags (60 00 00 01)
    struct.pack_into('<I',  block, 208, 0x01000060)

    # Bytes 212-219: brand (8 bytes, null-padded)
    brand = b'SpecScpe'
    block[212:220] = brand

    # Bytes 224-243: description (null-padded to 20 bytes)
    desc = b'Auto EQ Overlay'
    block[224:224 + len(desc)] = desc

    return bytes(block)


# ── Directory builder ─────────────────────────────────────────────────────────

def _dir_entry(name: str, size: int, offset: int) -> bytes:
    """Build one 40-byte directory entry."""
    name_bytes = name.encode('ascii')[:31]
    padded = name_bytes + b'\x00' * (32 - len(name_bytes))
    return padded + struct.pack('<II', size, offset)


# ── Public API ────────────────────────────────────────────────────────────────

def write_ovl(bands: list, name: str, path: str) -> None:
    """
    Write a Lake Controller .ovl file.

    bands : list of dicts from eq_suggester.suggest_eq()
            each: {'freq': Hz, 'gain': dB, 'q': float, 'type': str}
    name  : overlay display name in Lake Controller
    path  : destination file path (should end in .ovl)
    """
    if not bands:
        raise ValueError("No EQ bands to export.")

    # Build data blocks
    overlay_data  = _build_single_overlay(name)
    segments_data = b''.join(_build_band_block(i, b) for i, b in enumerate(bands))

    # Fixed data blocks in order
    data_blocks = [
        ('IOFileRev',                     _IO_FILE_REV),
        ('ControllerAppRev',              _CTRL_APP_REV),
        ('MinimumSupportedControllerRev', _MIN_CTRL_REV),
        ('MinimumSupportedControllerName',_MIN_CTRL_NAME),
        ('SubSystem',                     _SUBSYSTEM),
        ('ShowKey0',                      _SHOW_KEY0),
        ('SingleOverlay',                 overlay_data),
        ('SingleSegments',                segments_data),
        ('IOFileRevEndRev5_7',            _IO_REV_END),
        ('LakeEndOfFile',                 _EOF_MARKER),
    ]

    # Calculate offsets: header = 4 (magic) + 10 entries × 40 bytes = 404 bytes
    header_size = 4 + len(data_blocks) * 40
    offset = header_size
    offsets = []
    for _, block_data in data_blocks:
        offsets.append(offset)
        offset += len(block_data)

    # Assemble file
    output = bytearray()
    output += _FILE_MAGIC

    for (entry_name, block_data), off in zip(data_blocks, offsets):
        output += _dir_entry(entry_name, len(block_data), off)

    for _, block_data in data_blocks:
        output += block_data

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(output)
