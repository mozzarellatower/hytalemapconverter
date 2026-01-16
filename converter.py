#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import struct
import gzip
import zlib
import multiprocessing
import time
import sys
from collections import defaultdict, deque
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
)

import nbtlib
import zstandard as zstd
from bson import BSON, Binary

MAGIC = b"HytaleIndexedStorage"
BLOCKS_PER_SECTION = 32 * 32 * 32
HALF_BYTE_BLOCKS_LEN = BLOCKS_PER_SECTION // 2
BYTE_BLOCKS_LEN = BLOCKS_PER_SECTION

PALETTE_EMPTY = 0
PALETTE_HALF_BYTE = 1
PALETTE_BYTE = 2
PALETTE_SHORT = 3

WORLD_CACHE_MAGIC = b"HYTC"
WORLD_CACHE_VERSION = 1


def read_region_header(path):
    with open(path, "rb") as f:
        magic = f.read(20)
        if magic != MAGIC:
            raise ValueError(f"{path} does not start with HytaleIndexedStorage magic")
        version, blob_count, segment_size = struct.unpack(">III", f.read(12)
        )
        indexes = list(struct.unpack(">" + "I" * blob_count, f.read(blob_count * 4)))
    return version, blob_count, segment_size, indexes


def read_region_blob(path, start_segment, segment_size, blob_count):
    if start_segment == 0:
        return None
    segments_base = 20 + 12 + blob_count * 4
    offset = segments_base + (start_segment - 1) * segment_size
    with open(path, "rb") as f:
        f.seek(offset)
        header = f.read(8)
        if len(header) < 8:
            return None
        uncompressed_size = struct.unpack(">I", header[:4])[0]
        compressed_size = struct.unpack(">I", header[4:8])[0]
        total = 8 + compressed_size
        segments = (total + segment_size - 1) // segment_size
        f.seek(offset)
        blob = f.read(segments * segment_size)
    compressed = blob[8 : 8 + compressed_size]
    data = zstd.ZstdDecompressor().decompress(
        compressed, max_output_size=uncompressed_size
    )
    return data


def write_region_file(path, version, blob_count, segment_size, blobs):
    indexes = [0] * blob_count
    segment_data = []
    next_segment = 1

    for idx, blob in blobs.items():
        if blob is None:
            continue
        compressed = zstd.ZstdCompressor(level=3).compress(blob)
        header = struct.pack(">I", len(blob)) + struct.pack(">I", len(compressed))
        payload = header + compressed
        segments_needed = (len(payload) + segment_size - 1) // segment_size
        indexes[idx] = next_segment
        # pad to full segment size
        padded_len = segments_needed * segment_size
        payload += b"\x00" * (padded_len - len(payload))
        segment_data.append(payload)
        next_segment += segments_needed

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">III", version, blob_count, segment_size))
        f.write(struct.pack(">" + "I" * blob_count, *indexes))
        for payload in segment_data:
            f.write(payload)


def list_region_files(mc_world):
    region_dir = os.path.join(mc_world, "region")
    if not os.path.isdir(region_dir):
        raise FileNotFoundError(f"No region/ folder found in {mc_world}")
    region_files = [
        os.path.join(region_dir, filename)
        for filename in os.listdir(region_dir)
        if filename.endswith(".mca")
    ]
    return sorted(region_files)


def estimate_world_size_bytes(mc_world):
    if os.path.isfile(mc_world):
        return os.path.getsize(mc_world)
    total = 0
    for region_path in list_region_files(mc_world):
        total += os.path.getsize(region_path)
    return total


def parse_region_filename(filename):
    # Expected format: r.<x>.<z>.mca
    name = os.path.basename(filename)
    if not (name.startswith("r.") and name.endswith(".mca")):
        raise ValueError(f"Invalid region filename: {filename}")
    parts = name.split(".")
    if len(parts) != 4:
        raise ValueError(f"Invalid region filename: {filename}")
    return int(parts[1]), int(parts[2])


def atomic_write_json(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp_path, path)


def read_mc_chunk_bytes(region_path, chunk_index):
    with open(region_path, "rb") as f:
        f.seek(chunk_index * 4)
        entry = struct.unpack(">I", f.read(4))[0]
        sector_offset = entry >> 8
        sector_count = entry & 0xFF
        if sector_offset == 0:
            return None
        f.seek(sector_offset * 4096)
        length = struct.unpack(">I", f.read(4))[0]
        compression = struct.unpack(">B", f.read(1))[0]
        data = f.read(length - 1)
    if compression == 1:
        return gzip.decompress(data)
    if compression == 2:
        return zlib.decompress(data)
    return None


def iter_mc_chunks(region_path):
    with open(region_path, "rb") as f:
        header = f.read(4096)
    if len(header) < 4096:
        print(
            f"Warning: skipping invalid region file (short header): {region_path}"
        )
        return
    for i in range(1024):
        entry = struct.unpack(">I", header[i * 4 : (i + 1) * 4])[0]
        sector_offset = entry >> 8
        if sector_offset == 0:
            continue
        data = read_mc_chunk_bytes(region_path, i)
        if not data:
            continue
        nbt = nbtlib.File.parse(io.BytesIO(data))
        root = nbt["Level"] if "Level" in nbt else nbt
        chunk_x = int(root.get("xPos", 0))
        chunk_z = int(root.get("zPos", 0))
        sections = root.get("Sections")
        if sections is None:
            sections = root.get("sections", [])
        yield chunk_x, chunk_z, sections


def write_chunk_cache(cache_path, entries):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(WORLD_CACHE_MAGIC)
        f.write(struct.pack(">HI", WORLD_CACHE_VERSION, len(entries)))
        for chunk_x, chunk_z, blob in entries:
            f.write(struct.pack(">iiI", chunk_x, chunk_z, len(blob)))
            f.write(blob)


def iter_chunk_cache(cache_path, load_blobs=True):
    with open(cache_path, "rb") as f:
        magic = f.read(4)
        if magic != WORLD_CACHE_MAGIC:
            raise ValueError(f"Invalid cache file: {cache_path}")
        version, count = struct.unpack(">HI", f.read(6))
        if version != WORLD_CACHE_VERSION:
            raise ValueError(
                f"Unsupported cache version {version} in {cache_path}"
            )
        for _ in range(count):
            chunk_x, chunk_z, size = struct.unpack(">iiI", f.read(12))
            if load_blobs:
                blob = f.read(size)
                yield chunk_x, chunk_z, blob
            else:
                f.seek(size, os.SEEK_CUR)
                yield chunk_x, chunk_z, None


def build_chunk_blob(builder, template_info):
    sections_out = builder.build_sections(
        template_info.asset_count,
        template_info.empty_fluid_data,
        template_info.block_physics_data,
    )
    chunk_doc = {
        "Components": {
            "BlockComponentChunk": {"BlockComponents": {}},
            "EnvironmentChunk": {"Data": Binary(template_info.environment_data)},
            "ChunkColumn": {"Sections": sections_out},
            "WorldChunk": {},
            "BlockHealthChunk": {"Data": Binary(template_info.block_health_data)},
            "BlockChunk": {
                "Version": template_info.block_chunk_version,
                "Data": Binary(template_info.block_chunk_data),
            },
            "EntityChunk": {"Entities": []},
        }
    }
    return BSON.encode(chunk_doc)


def build_region_chunk_blobs(region_path, mapper, template_info):
    chunks = {}
    unknown_counts = defaultdict(int)
    converted_blocks = 0

    for chunk_x, chunk_z, sections in iter_mc_chunks(region_path):
        for sec in sections:
            sec_y = int(sec.get("Y", 0))
            blocks = sec.get("Blocks")
            data = sec.get("Data")
            palette, block_states = get_section_palette_and_states(sec)

            if blocks is not None and data is not None:
                add = sec.get("Add")
                blocks_bytes = bytes(blocks)
                data_bytes = bytes(data)
                add_bytes = bytes(add) if add is not None else None

                for idx, block_id in enumerate(blocks_bytes):
                    nibble = data_bytes[idx // 2]
                    if idx % 2 == 0:
                        block_data = nibble & 0x0F
                    else:
                        block_data = (nibble >> 4) & 0x0F

                    if add_bytes is not None:
                        add_nibble = add_bytes[idx // 2]
                        if idx % 2 == 0:
                            block_id |= (add_nibble & 0x0F) << 8
                        else:
                            block_id |= (add_nibble >> 4) << 8

                    if block_id == 0 and block_data == 0:
                        continue

                    if not mapper.is_mapped_legacy(block_id, block_data):
                        unknown_counts[(block_id, block_data)] += 1

                    block_name = mapper.map_legacy(block_id, block_data)
                    if block_name == "Empty":
                        continue

                    x = idx & 0x0F
                    z = (idx >> 4) & 0x0F
                    y = (idx >> 8) & 0x0F

                    global_x = chunk_x * 16 + x
                    global_z = chunk_z * 16 + z
                    global_y = sec_y * 16 + y

                    h_chunk_x, h_local_x = floor_divmod(global_x, 32)
                    h_chunk_z, h_local_z = floor_divmod(global_z, 32)

                    chunk_key = (h_chunk_x, h_chunk_z)
                    if chunk_key not in chunks:
                        chunks[chunk_key] = HytaleChunkBuilder(
                            template_info.section_count
                        )
                    chunks[chunk_key].set_block(
                        h_local_x, global_y, h_local_z, block_name
                    )
                    converted_blocks += 1
            elif palette is not None and block_states is not None:
                palette_list = list(palette)
                indices = decode_block_states(block_states, len(palette_list))
                for idx, palette_index in enumerate(indices):
                    if palette_index >= len(palette_list):
                        continue
                    entry = palette_list[palette_index]
                    state_key = build_state_key(entry)
                    if state_key in (
                        "minecraft:air",
                        "minecraft:cave_air",
                        "minecraft:void_air",
                        "air",
                        "cave_air",
                        "void_air",
                    ):
                        continue
                    block_name = mapper.map_modern(state_key)
                    if block_name == "Empty":
                        continue

                    x = idx & 0x0F
                    z = (idx >> 4) & 0x0F
                    y = (idx >> 8) & 0x0F

                    global_x = chunk_x * 16 + x
                    global_z = chunk_z * 16 + z
                    global_y = sec_y * 16 + y

                    h_chunk_x, h_local_x = floor_divmod(global_x, 32)
                    h_chunk_z, h_local_z = floor_divmod(global_z, 32)

                    chunk_key = (h_chunk_x, h_chunk_z)
                    if chunk_key not in chunks:
                        chunks[chunk_key] = HytaleChunkBuilder(
                            template_info.section_count
                        )
                    chunks[chunk_key].set_block(
                        h_local_x, global_y, h_local_z, block_name
                    )
                    converted_blocks += 1

    entries = []
    for (chunk_x, chunk_z), builder in chunks.items():
        blob = build_chunk_blob(builder, template_info)
        entries.append((chunk_x, chunk_z, blob))
    return entries, converted_blocks, unknown_counts, len(entries) == 0


_WORKER_TEMPLATE_INFO = None
_WORKER_MAPPER = None


def _init_region_worker(template_cache, mapping_path, default_block):
    global _WORKER_TEMPLATE_INFO, _WORKER_MAPPER
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    _WORKER_TEMPLATE_INFO = TemplateInfo.from_cache(template_cache)
    _WORKER_MAPPER = BlockMapper(mapping_path, default_block=default_block)


def _process_region_worker(region_path, cache_path):
    msg = (
        f"Worker {os.getpid()} caching: {os.path.basename(region_path)}\n"
    ).encode("utf-8")
    try:
        os.write(1, msg)
    except OSError:
        print(
            msg.decode("utf-8", errors="replace").rstrip(),
            flush=True,
        )
    entries, converted_blocks, unknown_counts, is_empty = build_region_chunk_blobs(
        region_path, _WORKER_MAPPER, _WORKER_TEMPLATE_INFO
    )
    write_chunk_cache(cache_path, entries)
    return converted_blocks, unknown_counts, is_empty


def index_block(x, y, z):
    return ((y & 31) << 10) | ((z & 31) << 5) | (x & 31)


def write_utf(buf, s):
    data = s.encode("utf-8")
    if len(data) > 65535:
        raise ValueError(f"String too long for UTF: {s[:50]}")
    buf.extend(struct.pack(">H", len(data)))
    buf.extend(data)


def set_nibble(buf, idx, value):
    byte_index = idx >> 1
    if idx & 1:
        buf[byte_index] = (buf[byte_index] & 0x0F) | ((value & 0x0F) << 4)
    else:
        buf[byte_index] = (buf[byte_index] & 0xF0) | (value & 0x0F)


def build_palette(blocks, asset_count):
    # blocks: dict index->block_name (non-empty only)
    unique = set(blocks.values())
    if not unique:
        # all empty
        buf = bytearray()
        buf.extend(struct.pack(">I", asset_count))
        buf.append(PALETTE_EMPTY)
        # ticking bitset
        buf.extend(struct.pack(">H", 0))
        buf.extend(struct.pack(">H", 0))
        # filler palette type
        buf.append(PALETTE_EMPTY)
        # rotation palette type
        buf.append(PALETTE_EMPTY)
        # local light (short + boolean)
        buf.extend(struct.pack(">H", 0))
        buf.append(0)
        # global light
        buf.extend(struct.pack(">H", 0))
        buf.append(0)
        # counters
        buf.extend(struct.pack(">H", 0))
        buf.extend(struct.pack(">H", 0))
        return bytes(buf)

    unique.add("Empty")

    if len(unique) <= 16:
        palette_type = PALETTE_HALF_BYTE
        blocks_len = HALF_BYTE_BLOCKS_LEN
    elif len(unique) <= 256:
        palette_type = PALETTE_BYTE
        blocks_len = BYTE_BLOCKS_LEN
    else:
        raise ValueError(f"Too many unique blocks in section: {len(unique)}")

    # assign internal IDs, keep Empty at 0
    names = ["Empty"] + sorted(name for name in unique if name != "Empty")
    internal_by_name = {name: idx for idx, name in enumerate(names)}

    # build blocks array
    if palette_type == PALETTE_HALF_BYTE:
        fill = internal_by_name["Empty"] & 0x0F
        fill_byte = fill | (fill << 4)
        block_bytes = bytearray([fill_byte]) * blocks_len
    else:
        fill = internal_by_name["Empty"] & 0xFF
        block_bytes = bytearray([fill]) * blocks_len

    counts = defaultdict(int)
    for idx, name in blocks.items():
        internal_id = internal_by_name[name]
        counts[internal_id] += 1
        if palette_type == PALETTE_HALF_BYTE:
            set_nibble(block_bytes, idx, internal_id)
        else:
            block_bytes[idx] = internal_id

    empty_id = internal_by_name["Empty"]
    counts[empty_id] = BLOCKS_PER_SECTION - sum(
        counts[k] for k in counts if k != empty_id
    )

    buf = bytearray()
    buf.extend(struct.pack(">I", asset_count))
    buf.append(palette_type)
    buf.extend(struct.pack(">H", len(names)))

    for internal_id, name in enumerate(names):
        buf.append(internal_id & 0xFF)
        write_utf(buf, name)
        buf.extend(struct.pack(">H", counts.get(internal_id, 0)))

    buf.extend(block_bytes)

    # ticking bitset
    buf.extend(struct.pack(">H", 0))
    buf.extend(struct.pack(">H", 0))

    # filler palette type
    buf.append(PALETTE_EMPTY)
    # rotation palette type
    buf.append(PALETTE_EMPTY)

    # local light
    buf.extend(struct.pack(">H", 0))
    buf.append(0)

    # global light
    buf.extend(struct.pack(">H", 0))
    buf.append(0)

    # counters
    buf.extend(struct.pack(">H", 0))
    buf.extend(struct.pack(">H", 0))

    return bytes(buf)


class BlockMapper:
    def __init__(self, mapping_path=None, default_block=None):
        self.mapping = {}
        self.legacy = {}
        self.legacy_by_id = {}
        self.default = "Rock_Stone"
        if mapping_path:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.mapping = data.get("modern", {})
            self.legacy = data.get("legacy", {})
            self.legacy_by_id = data.get("legacy_by_id", {})
            self.default = data.get("default", self.default)
        if default_block:
            self.default = normalize_default_block(default_block)

    def map_modern(self, name):
        if name in self.mapping:
            return self.mapping[name]
        if name.startswith("minecraft:"):
            short = name.split(":", 1)[1]
            if short in self.mapping:
                return self.mapping[short]
        return self.default

    def map_legacy(self, block_id, block_data):
        key = f"{block_id}:{block_data}"
        if key in self.legacy:
            return self.legacy[key]
        if str(block_id) in self.legacy_by_id:
            return self.legacy_by_id[str(block_id)]
        return self.default

    def is_mapped_legacy(self, block_id, block_data):
        key = f"{block_id}:{block_data}"
        return key in self.legacy or str(block_id) in self.legacy_by_id


def normalize_default_block(value):
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in ("air", "empty"):
        return "Empty"
    return value


class TemplateInfo:
    def __init__(self, template_world):
        chunks_dir = os.path.join(template_world, "chunks")
        region_files = [
            f
            for f in os.listdir(chunks_dir)
            if f.endswith(".region.bin")
        ]
        if not region_files:
            raise FileNotFoundError("No region files found in template world")
        region_path = os.path.join(chunks_dir, sorted(region_files)[0])
        self.region_path = region_path
        self.version, self.blob_count, self.segment_size, indexes = read_region_header(
            region_path
        )
        # pick first non-empty chunk as template
        start_segment = next(i for i in indexes if i != 0)
        blob = read_region_blob(
            region_path, start_segment, self.segment_size, self.blob_count
        )
        if not blob:
            raise RuntimeError("Failed to read template chunk")
        doc = BSON(blob).decode()
        components = doc["Components"]

        sections = components["ChunkColumn"]["Sections"]
        self.section_count = len(sections)

        # asset count from first section block data
        first_section_block = sections[0]["Components"]["Block"]["Data"]
        self.asset_count = struct.unpack(">I", first_section_block[:4])[0]

        self.block_chunk_version = components["BlockChunk"].get("Version", 6)
        self.block_chunk_data = components["BlockChunk"]["Data"]
        self.block_health_data = components["BlockHealthChunk"]["Data"]
        self.environment_data = components["EnvironmentChunk"]["Data"]

        # empty fluid data from smallest sample
        fluid_samples = [
            s["Components"]["Fluid"]["Data"]
            for s in sections
            if "Fluid" in s["Components"]
        ]
        self.empty_fluid_data = min(fluid_samples, key=len)
        self.block_physics_data = b"\x00"

    @classmethod
    def from_cache(cls, cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        info = cls.__new__(cls)
        info.version = payload["version"]
        info.blob_count = payload["blob_count"]
        info.segment_size = payload["segment_size"]
        info.section_count = payload["section_count"]
        info.asset_count = payload["asset_count"]
        info.block_chunk_version = payload["block_chunk_version"]
        info.block_chunk_data = base64.b64decode(payload["block_chunk_data"])
        info.block_health_data = base64.b64decode(payload["block_health_data"])
        info.environment_data = base64.b64decode(payload["environment_data"])
        info.empty_fluid_data = base64.b64decode(payload["empty_fluid_data"])
        info.block_physics_data = base64.b64decode(payload["block_physics_data"])
        info.region_path = None
        return info

    def save_cache(self, cache_path):
        payload = {
            "version": self.version,
            "blob_count": self.blob_count,
            "segment_size": self.segment_size,
            "section_count": self.section_count,
            "asset_count": self.asset_count,
            "block_chunk_version": self.block_chunk_version,
            "block_chunk_data": base64.b64encode(self.block_chunk_data).decode("ascii"),
            "block_health_data": base64.b64encode(self.block_health_data).decode(
                "ascii"
            ),
            "environment_data": base64.b64encode(self.environment_data).decode("ascii"),
            "empty_fluid_data": base64.b64encode(self.empty_fluid_data).decode("ascii"),
            "block_physics_data": base64.b64encode(
                self.block_physics_data
            ).decode("ascii"),
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


class HytaleChunkBuilder:
    def __init__(self, section_count):
        self.section_count = section_count
        self.sections = [defaultdict(str) for _ in range(section_count)]

    def set_block(self, x, y, z, block_name):
        section_index = y // 32
        if section_index < 0 or section_index >= self.section_count:
            return
        local_y = y % 32
        idx = index_block(x, local_y, z)
        if block_name != "Empty":
            self.sections[section_index][idx] = block_name

    def build_sections(self, asset_count, empty_fluid_data, block_physics_data):
        sections_out = []
        for section_blocks in self.sections:
            block_data = build_palette(section_blocks, asset_count)
            components = {
                "ChunkSection": {},
                "Block": {"Version": 6, "Data": Binary(block_data)},
                "Fluid": {"Data": Binary(empty_fluid_data)},
                "BlockPhysics": {"Data": Binary(block_physics_data)},
            }
            sections_out.append({"Components": components})
        return sections_out


def floor_divmod(value, divisor):
    q = value // divisor
    r = value % divisor
    return q, r


def build_state_key(palette_entry):
    name = palette_entry.get("Name") or palette_entry.get("name", "")
    properties = palette_entry.get("Properties")
    if properties is None:
        properties = palette_entry.get("properties")
    if not properties:
        return name
    parts = []
    for key in sorted(properties.keys()):
        parts.append(f"{key}={properties[key]}")
    return f"{name}[{','.join(parts)}]"


def get_section_palette_and_states(section):
    palette = section.get("Palette")
    block_states = section.get("BlockStates")
    if palette is None or block_states is None:
        block_states_comp = section.get("block_states")
        if block_states_comp is not None:
            palette = block_states_comp.get("palette")
            block_states = block_states_comp.get("data")
    return palette, block_states


def decode_block_states(block_states, palette_size):
    if palette_size <= 1:
        return [0] * 4096
    bits_per_block = max(4, (palette_size - 1).bit_length())
    mask = (1 << bits_per_block) - 1
    longs = [value & 0xFFFFFFFFFFFFFFFF for value in block_states]
    indices = []
    for i in range(4096):
        bit_index = i * bits_per_block
        long_index = bit_index >> 6
        start_bit = bit_index & 63
        value = (longs[long_index] >> start_bit) & mask
        if start_bit + bits_per_block > 64:
            value |= (longs[long_index + 1] << (64 - start_bit)) & mask
        indices.append(value)
    return indices


def prepare_output_world(output_world, template):
    os.makedirs(output_world, exist_ok=True)
    os.makedirs(os.path.join(output_world, "chunks"), exist_ok=True)
    if template:
        for name in ("config.json", "config.json.bak"):
            src = os.path.join(template, name)
            if os.path.exists(src):
                with open(src, "rb") as fsrc, open(
                    os.path.join(output_world, name), "wb"
                ) as fdst:
                    fdst.write(fsrc.read())
        resources_src = os.path.join(template, "resources")
        resources_dst = os.path.join(output_world, "resources")
        if os.path.isdir(resources_src) and not os.path.exists(resources_dst):
            os.makedirs(resources_dst, exist_ok=True)
            for filename in os.listdir(resources_src):
                src = os.path.join(resources_src, filename)
                dst = os.path.join(resources_dst, filename)
                if os.path.isfile(src):
                    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                        fdst.write(fsrc.read())
    else:
        print("No template provided; skipping config/resources copy.")


def convert_world(
    mc_world, output_world, template, mapping_path, template_cache=None, default_block=None
):
    mapper = BlockMapper(mapping_path, default_block=default_block)
    template_info = None
    if template_cache and os.path.exists(template_cache):
        template_info = TemplateInfo.from_cache(template_cache)
    if template_info is None and template:
        template_info = TemplateInfo(template)
        if template_cache:
            template_info.save_cache(template_cache)
    if template_info is None:
        raise ValueError("template or template cache is required")

    # build chunk builders
    chunks = {}
    unknown_counts = defaultdict(int)
    converted_blocks = 0

    region_dir = os.path.join(mc_world, "region")
    for filename in os.listdir(region_dir):
        if not filename.endswith(".mca"):
            continue
        region_has_blocks = False
        region_path = os.path.join(region_dir, filename)
        for chunk_x, chunk_z, sections in iter_mc_chunks(region_path):
            for sec in sections:
                sec_y = int(sec.get("Y", 0))
                blocks = sec.get("Blocks")
                data = sec.get("Data")
                palette, block_states = get_section_palette_and_states(sec)

                if blocks is not None and data is not None:
                    add = sec.get("Add")
                    blocks_bytes = bytes(blocks)
                    data_bytes = bytes(data)
                    add_bytes = bytes(add) if add is not None else None

                    for idx, block_id in enumerate(blocks_bytes):
                        # compute data nibble
                        nibble = data_bytes[idx // 2]
                        if idx % 2 == 0:
                            block_data = nibble & 0x0F
                        else:
                            block_data = (nibble >> 4) & 0x0F

                        if add_bytes is not None:
                            add_nibble = add_bytes[idx // 2]
                            if idx % 2 == 0:
                                block_id |= (add_nibble & 0x0F) << 8
                            else:
                                block_id |= (add_nibble >> 4) << 8

                        if block_id == 0 and block_data == 0:
                            continue

                        if not mapper.is_mapped_legacy(block_id, block_data):
                            unknown_counts[(block_id, block_data)] += 1

                        block_name = mapper.map_legacy(block_id, block_data)
                        if block_name == "Empty":
                            continue

                        # local mc coords
                        x = idx & 0x0F
                        z = (idx >> 4) & 0x0F
                        y = (idx >> 8) & 0x0F

                        global_x = chunk_x * 16 + x
                        global_z = chunk_z * 16 + z
                        global_y = sec_y * 16 + y

                        h_chunk_x, h_local_x = floor_divmod(global_x, 32)
                        h_chunk_z, h_local_z = floor_divmod(global_z, 32)

                        chunk_key = (h_chunk_x, h_chunk_z)
                        if chunk_key not in chunks:
                            chunks[chunk_key] = HytaleChunkBuilder(
                                template_info.section_count
                            )
                        chunks[chunk_key].set_block(
                            h_local_x, global_y, h_local_z, block_name
                        )
                        converted_blocks += 1
                        region_has_blocks = True
                elif palette is not None and block_states is not None:
                    palette_list = list(palette)
                    indices = decode_block_states(block_states, len(palette_list))
                    for idx, palette_index in enumerate(indices):
                        if palette_index >= len(palette_list):
                            continue
                        entry = palette_list[palette_index]
                        state_key = build_state_key(entry)
                        if state_key in (
                            "minecraft:air",
                            "minecraft:cave_air",
                            "minecraft:void_air",
                            "air",
                            "cave_air",
                            "void_air",
                        ):
                            continue
                        block_name = mapper.map_modern(state_key)
                        if block_name == "Empty":
                            continue

                        x = idx & 0x0F
                        z = (idx >> 4) & 0x0F
                        y = (idx >> 8) & 0x0F

                        global_x = chunk_x * 16 + x
                        global_z = chunk_z * 16 + z
                        global_y = sec_y * 16 + y

                        h_chunk_x, h_local_x = floor_divmod(global_x, 32)
                        h_chunk_z, h_local_z = floor_divmod(global_z, 32)

                        chunk_key = (h_chunk_x, h_chunk_z)
                        if chunk_key not in chunks:
                            chunks[chunk_key] = HytaleChunkBuilder(
                                template_info.section_count
                            )
                        chunks[chunk_key].set_block(
                            h_local_x, global_y, h_local_z, block_name
                        )
                        converted_blocks += 1
                        region_has_blocks = True
        if region_has_blocks:
            print(f"Region converted: {filename}")
        else:
            print(f"Region converted (empty): {filename}")
    prepare_output_world(output_world, template)

    # build region blobs
    region_blobs = defaultdict(dict)
    for (chunk_x, chunk_z), builder in chunks.items():
        blob = build_chunk_blob(builder, template_info)

        region_x, local_x = floor_divmod(chunk_x, 32)
        region_z, local_z = floor_divmod(chunk_z, 32)
        blob_index = local_x + local_z * 32
        region_blobs[(region_x, region_z)][blob_index] = blob

    for (region_x, region_z), blobs in region_blobs.items():
        region_name = f"{region_x}.{region_z}.region.bin"
        out_path = os.path.join(output_world, "chunks", region_name)
        write_region_file(
            out_path,
            template_info.version,
            template_info.blob_count,
            template_info.segment_size,
            blobs,
        )

    print(f"Converted {converted_blocks} blocks into {len(chunks)} chunks.")
    if unknown_counts:
        print("Unmapped legacy blocks (id:data -> count):")
        for (block_id, block_data), count in sorted(
            unknown_counts.items(), key=lambda item: -item[1]
        )[:20]:
            print(f"  {block_id}:{block_data} -> {count}")


def cache_path_for_region(cache_dir, region_filename):
    region_x, region_z = parse_region_filename(region_filename)
    return os.path.join(cache_dir, f"region_{region_x}_{region_z}.cache")


def load_worldcache_state(cache_dir):
    path = os.path.join(cache_dir, "worldcache.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_worldcache_state(cache_dir, state):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "worldcache.json")
    atomic_write_json(path, state)


def init_worldcache_state(
    mc_world,
    output_world,
    template,
    mapping_path,
    template_cache,
    default_block,
    mode,
    workers,
    region_files,
):
    return {
        "version": WORLD_CACHE_VERSION,
        "input": mc_world,
        "output": output_world,
        "template": template,
        "template_cache": template_cache,
        "mapping": mapping_path,
        "default_block": default_block,
        "mode": mode,
        "workers": workers,
        "region_files": region_files,
        "processed_regions": [],
        "phase": "cache",
        "converted_blocks": 0,
    }


def cleanup_worldcache(cache_dir, state):
    for region_filename in state.get("region_files", []):
        cache_path = cache_path_for_region(cache_dir, region_filename)
        if os.path.exists(cache_path):
            os.remove(cache_path)
    cache_path = os.path.join(cache_dir, "worldcache.json")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    if os.path.isdir(cache_dir) and not os.listdir(cache_dir):
        os.rmdir(cache_dir)


def convert_world_cached(state, cache_dir, ignore_prompt=False):
    mc_world = state["input"]
    output_world = state["output"]
    template = state["template"]
    mapping_path = state["mapping"]
    template_cache = state["template_cache"]
    default_block = state.get("default_block")
    mode = state.get("mode", "chunked")
    workers = state.get("workers", 1)

    if not template_cache or not os.path.exists(template_cache):
        raise ValueError("template cache is required for cached conversion")

    region_dir = os.path.join(mc_world, "region")
    if not os.path.isdir(region_dir):
        raise FileNotFoundError(f"No region/ folder found in {mc_world}")

    region_files = list(state.get("region_files") or [])
    if not region_files:
        region_files = [
            os.path.basename(path) for path in list_region_files(mc_world)
        ]
        state["region_files"] = region_files
        save_worldcache_state(cache_dir, state)

    processed = set(state.get("processed_regions", []))
    remaining = [name for name in region_files if name not in processed]
    print(
        f"Cache phase remaining regions: {len(remaining)}",
        flush=True,
    )

    total_converted = state.get("converted_blocks", 0)
    unknown_counts = defaultdict(int)

    if state.get("phase") == "cache" and remaining:
        total_regions = len(region_files)
        start_cache = time.monotonic()
        completed_in_run = 0
        first_region = remaining[0]
        print(
            "Caching regions "
            f"({len(remaining)}/{total_regions}), first region: {first_region}",
            flush=True,
        )
        os.makedirs(cache_dir, exist_ok=True)
        if mode in ("parallel", "parallel-batch") and workers and workers > 1:
            max_workers = workers or (os.cpu_count() or 1)
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_region_worker,
                initargs=(template_cache, mapping_path, default_block),
            ) as executor:
                futures = {}
                submit_times = {}
                print(
                    f"Queued {len(remaining)} regions across {max_workers} workers.",
                    flush=True,
                )

                def submit_region(region_filename):
                    region_path = os.path.join(region_dir, region_filename)
                    cache_path = cache_path_for_region(cache_dir, region_filename)
                    future = executor.submit(
                        _process_region_worker, region_path, cache_path
                    )
                    futures[future] = region_filename
                    submit_times[future] = time.monotonic()
                    print(f"Queued region: {region_filename}", flush=True)

                def handle_future(future, now):
                    nonlocal completed_in_run, total_converted
                    region_filename = futures[future]
                    converted_blocks, region_unknowns, is_empty = future.result()
                    total_converted += converted_blocks
                    for key, count in region_unknowns.items():
                        unknown_counts[key] += count
                    processed.add(region_filename)
                    state["processed_regions"] = sorted(processed)
                    state["converted_blocks"] = total_converted
                    save_worldcache_state(cache_dir, state)
                    completed_in_run += 1
                    elapsed = now - submit_times.get(future, start_cache)
                    elapsed_total = now - start_cache
                    rate = completed_in_run / elapsed_total if elapsed_total > 0 else 0
                    remaining_count = len(remaining) - completed_in_run
                    eta = remaining_count / rate if rate > 0 else 0
                    if completed_in_run == 1:
                        estimate = (
                            elapsed * len(remaining) / completed_in_run
                            if elapsed > 0
                            else 0
                        )
                        print(
                            "First region cached in "
                            f"{format_duration(elapsed)}; "
                            f"estimated cache time: {format_duration(estimate)}",
                            flush=True,
                        )
                    status = "Region cached (empty)" if is_empty else "Region cached"
                    print(
                        f"{status}: {region_filename} in {format_duration(elapsed)} "
                        f"({completed_in_run}/{len(remaining)} done, "
                        f"ETA {format_duration(eta)})",
                        flush=True,
                    )

                if mode == "parallel-batch":
                    region_queue = deque(remaining)
                    for _ in range(min(max_workers, len(region_queue))):
                        submit_region(region_queue.popleft())
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            now = time.monotonic()
                            handle_future(future, now)
                            del futures[future]
                            if region_queue:
                                submit_region(region_queue.popleft())
                else:
                    for region_filename in remaining:
                        submit_region(region_filename)
                    for future in as_completed(futures):
                        now = time.monotonic()
                        handle_future(future, now)
        else:
            mapper = BlockMapper(mapping_path, default_block=default_block)
            template_info = TemplateInfo.from_cache(template_cache)
            for idx, region_filename in enumerate(remaining, start=1):
                region_start = time.monotonic()
                print(
                    "Processing region "
                    f"{idx}/{total_regions}: {region_filename}",
                    flush=True,
                )
                region_path = os.path.join(region_dir, region_filename)
                cache_path = cache_path_for_region(cache_dir, region_filename)
                entries, converted_blocks, region_unknowns, is_empty = (
                    build_region_chunk_blobs(region_path, mapper, template_info)
                )
                write_chunk_cache(cache_path, entries)
                total_converted += converted_blocks
                for key, count in region_unknowns.items():
                    unknown_counts[key] += count
                processed.add(region_filename)
                state["processed_regions"] = sorted(processed)
                state["converted_blocks"] = total_converted
                save_worldcache_state(cache_dir, state)
                elapsed = time.monotonic() - region_start
                if idx == 1:
                    estimate = elapsed * len(remaining)
                    print(
                        "First region cached in "
                        f"{format_duration(elapsed)}; "
                        f"estimated cache time: {format_duration(estimate)}"
                    )
                if is_empty:
                    print(
                        f"Region cached (empty): {region_filename} "
                        f"in {format_duration(elapsed)}"
                    )
                else:
                    print(
                        f"Region cached: {region_filename} "
                        f"in {format_duration(elapsed)}"
                    )

    if state.get("phase") == "cache":
        state["phase"] = "merge"
        save_worldcache_state(cache_dir, state)

    prepare_output_world(output_world, template)

    expected_counts = defaultdict(int)
    for region_filename in state.get("region_files", []):
        cache_path = cache_path_for_region(cache_dir, region_filename)
        if not os.path.exists(cache_path):
            continue
        for chunk_x, chunk_z, _ in iter_chunk_cache(
            cache_path, load_blobs=False
        ):
            region_x, local_x = floor_divmod(chunk_x, 32)
            region_z, local_z = floor_divmod(chunk_z, 32)
            expected_counts[(region_x, region_z)] += 1

    region_buffers = defaultdict(dict)
    region_counts = defaultdict(int)
    template_info = TemplateInfo.from_cache(template_cache)
    for region_filename in state.get("region_files", []):
        cache_path = cache_path_for_region(cache_dir, region_filename)
        if not os.path.exists(cache_path):
            continue
        for chunk_x, chunk_z, blob in iter_chunk_cache(cache_path, load_blobs=True):
            region_x, local_x = floor_divmod(chunk_x, 32)
            region_z, local_z = floor_divmod(chunk_z, 32)
            blob_index = local_x + local_z * 32
            region_buffers[(region_x, region_z)][blob_index] = blob
            region_counts[(region_x, region_z)] += 1
            if region_counts[(region_x, region_z)] >= expected_counts.get(
                (region_x, region_z), 0
            ):
                region_name = f"{region_x}.{region_z}.region.bin"
                out_path = os.path.join(output_world, "chunks", region_name)
                write_region_file(
                    out_path,
                    template_info.version,
                    template_info.blob_count,
                    template_info.segment_size,
                    region_buffers[(region_x, region_z)],
                )
                print(f"Region converted: {region_name}")
                del region_buffers[(region_x, region_z)]

    print(f"Converted {total_converted} blocks using cached conversion.")
    if unknown_counts:
        print("Unmapped legacy blocks (id:data -> count):")
        for (block_id, block_data), count in sorted(
            unknown_counts.items(), key=lambda item: -item[1]
        )[:20]:
            print(f"  {block_id}:{block_data} -> {count}")

    cleanup_worldcache(cache_dir, state)


def format_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def format_duration(seconds):
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def run_world_conversion(
    mc_world,
    output_world,
    template,
    mapping_path,
    template_cache=None,
    default_block=None,
    mode="auto",
    workers=None,
    cache_dir="worldcache",
    continue_mode=False,
    ignore_prompt=False,
    size_threshold_bytes=100 * 1024 * 1024,
):
    if continue_mode:
        state = load_worldcache_state(cache_dir)
        if not state:
            raise FileNotFoundError(
                f"No worldcache.json found in {cache_dir} to continue."
            )
        if not ignore_prompt:
            processed = len(state.get("processed_regions", []))
            total = len(state.get("region_files", []))
            phase = state.get("phase", "cache")
            print(f"Resume cached conversion: {processed}/{total} regions, phase={phase}.")
            choice = input("Continue? [Y/n] ").strip().lower()
            if choice and choice[0] == "n":
                print("Cancelled.")
                return
        convert_world_cached(state, cache_dir, ignore_prompt=ignore_prompt)
        return

    world_size = estimate_world_size_bytes(mc_world)
    chosen_mode = mode
    if chosen_mode == "auto":
        if world_size >= size_threshold_bytes:
            if ignore_prompt:
                chosen_mode = "parallel" if (workers and workers > 1) else "chunked"
            else:
                print(
                    f"World size is {format_size(world_size)}. "
                    "Choose conversion mode: [p]arallel or [c]hunked."
                )
                choice = input("Mode (p/c): ").strip().lower()
                if choice.startswith("p"):
                    chosen_mode = "parallel"
                elif choice.startswith("c"):
                    chosen_mode = "chunked"
                else:
                    print("Cancelled.")
                    return
        else:
            chosen_mode = "in-memory"

    if chosen_mode in ("chunked", "parallel", "parallel-batch"):
        if not template_cache:
            if not template:
                raise ValueError("template or template cache is required for cached conversion")
            template_cache = os.path.join(cache_dir, "template_cache.json")
            TemplateInfo(template).save_cache(template_cache)
        region_files = [
            os.path.basename(path) for path in list_region_files(mc_world)
        ]
        worker_count = workers or (os.cpu_count() or 1)
        if chosen_mode not in ("parallel", "parallel-batch"):
            worker_count = 1
        state = init_worldcache_state(
            mc_world,
            output_world,
            template,
            mapping_path,
            template_cache,
            default_block,
            chosen_mode,
            worker_count,
            region_files,
        )
        save_worldcache_state(cache_dir, state)
        convert_world_cached(state, cache_dir, ignore_prompt=ignore_prompt)
        return

    convert_world(
        mc_world,
        output_world,
        template,
        mapping_path,
        template_cache,
        default_block=default_block,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Minecraft map to a Hytale world format."
    )
    parser.add_argument(
        "--input",
        required=False,
        help="Path to Minecraft world folder (contains region/)",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Path to output Hytale world folder",
    )
    parser.add_argument(
        "--template",
        default=None,
        help=(
            "Path to template Hytale world folder "
            "(e.g., serverexample/universe/worlds/default)"
        ),
    )
    parser.add_argument(
        "--template-cache",
        default=None,
        help="Path to template cache JSON (used or created)",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional mapping JSON to override block mappings",
    )
    parser.add_argument(
        "--default-block",
        default=None,
        help="Override default block for unmapped entries (e.g., Empty or Air)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "in-memory", "chunked", "parallel", "parallel-batch"),
        default="auto",
        help="Conversion mode; auto prompts on large worlds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count for parallel mode (defaults to CPU count).",
    )
    parser.add_argument(
        "--cache-dir",
        default="worldcache",
        help="Directory to store cache files for large conversions.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_mode",
        action="store_true",
        help="Resume a cached conversion from worldcache.",
    )
    parser.add_argument(
        "--ignoreprompt",
        action="store_true",
        help="Skip prompts and use defaults where possible.",
    )

    args = parser.parse_args()
    if not args.continue_mode:
        if not args.input or not args.output:
            parser.error("--input and --output are required unless --continue is set.")
    mapping_path = args.mapping
    if mapping_path is None:
        default_mapping = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mappings", "default.json"
        )
        if os.path.exists(default_mapping):
            mapping_path = default_mapping
    template_cache = args.template_cache
    if template_cache is None and args.template is None:
        default_cache = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "template_cache.json"
        )
        if os.path.exists(default_cache):
            template_cache = default_cache

    run_world_conversion(
        args.input,
        args.output,
        args.template,
        mapping_path,
        template_cache,
        default_block=args.default_block,
        mode=args.mode,
        workers=args.workers,
        cache_dir=args.cache_dir,
        continue_mode=args.continue_mode,
        ignore_prompt=args.ignoreprompt,
    )


if __name__ == "__main__":
    main()
