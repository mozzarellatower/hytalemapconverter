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
import math
import hashlib
from collections import defaultdict, deque, Counter
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

FLUID_MAX_LEVEL_DEFAULT = 8
FLUID_MAX_LEVELS = {
    "Water": 8,
    "Water_Source": 1,
    "Water_Finite": 8,
    "Lava": 8,
    "Lava_Source": 1,
    "Poison": 8,
    "Poison_Source": 1,
    "Slime": 8,
    "Slime_Source": 1,
    "Slime_Red": 8,
    "Slime_Red_Source": 1,
    "Tar": 8,
    "Tar_Source": 1,
}
FLUID_LEGACY_NAME_MAP = {
    "Fluid_Water": ("Water", "Water_Source"),
    "Fluid_Water_Test": ("Water_Finite", None),
    "Fluid_Lava": ("Lava", "Lava_Source"),
    "Fluid_Tar": ("Tar", "Tar_Source"),
    "Fluid_Slime": ("Slime", "Slime_Source"),
    "Fluid_Slime_Red": ("Slime_Red", "Slime_Red_Source"),
    "Fluid_Poison": ("Poison", "Poison_Source"),
}

WORLD_CACHE_MAGIC = b"HYTC"
WORLD_CACHE_VERSION = 1

HEIGHTMAP_OFFSET = 32768
DEFAULT_FALLBACK_THRESHOLD = 5.0
SEAM_MISMATCH_THRESHOLD = 2


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


def _safe_mkdir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _try_import_pil():
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    return Image


def scan_mc_y_range(mc_world):
    min_section_y = None
    max_section_y = None
    region_dir = os.path.join(mc_world, "region")
    for region_path in list_region_files(mc_world):
        for _chunk_x, _chunk_z, sections in iter_mc_chunks(region_path):
            for sec in sections:
                sec_y = sec.get("Y")
                if sec_y is None:
                    sec_y = sec.get("y", 0)
                try:
                    sec_y = int(sec_y)
                except Exception:
                    sec_y = 0
                if min_section_y is None or sec_y < min_section_y:
                    min_section_y = sec_y
                if max_section_y is None or sec_y > max_section_y:
                    max_section_y = sec_y
    if min_section_y is None:
        min_section_y = 0
    if max_section_y is None:
        max_section_y = 0
    return min_section_y, max_section_y


def _material_to_color(name):
    if not name or name == "Empty":
        return (0, 0, 0)
    digest = hashlib.md5(name.encode("utf-8")).digest()
    # Bias towards mid tones so adjacent materials are visible.
    r = 64 + (digest[0] % 160)
    g = 64 + (digest[1] % 160)
    b = 64 + (digest[2] % 160)
    return (r, g, b)


def _write_pgm(path, width, height, values, maxval=65535):
    with open(path, "wb") as f:
        header = f"P5 {width} {height} {maxval}\n".encode("ascii")
        f.write(header)
        if maxval > 255:
            for value in values:
                f.write(struct.pack(">H", value))
        else:
            f.write(bytes(values))


def _write_ppm(path, width, height, values):
    with open(path, "wb") as f:
        header = f"P6 {width} {height} 255\n".encode("ascii")
        f.write(header)
        f.write(bytes(values))


def _write_heightmap_image(path, heightmap, image_lib=None):
    width = 16
    height = 16
    values = []
    for z in range(height):
        for x in range(width):
            y = heightmap[z][x]
            if y is None:
                values.append(0)
            else:
                value = max(0, min(65535, y + HEIGHTMAP_OFFSET))
                values.append(value)
    if image_lib:
        buf = bytearray()
        for value in values:
            buf.extend(struct.pack("<H", value))
        img = image_lib.frombytes("I;16", (width, height), bytes(buf))
        img.save(path)
    else:
        _write_pgm(path, width, height, values, maxval=65535)


def _write_surface_image(path, surface_material, image_lib=None):
    width = 16
    height = 16
    values = []
    for z in range(height):
        for x in range(width):
            material = surface_material[z][x]
            r, g, b = _material_to_color(material)
            values.extend((r, g, b))
    if image_lib:
        img = image_lib.new("RGB", (width, height))
        pixels = list(zip(values[0::3], values[1::3], values[2::3]))
        img.putdata(pixels)
        img.save(path)
    else:
        _write_ppm(path, width, height, values)


class ConversionStats:
    def __init__(self):
        self.block_count = 0
        self.fluid_count = 0
        self.fallback_blocks = 0
        self.fallback_keys = defaultdict(int)
        self.unknown_legacy = defaultdict(int)
        self.unknown_modern = defaultdict(int)
        self.bounds_errors = 0
        self.hytale_oob = 0
        self.y_oob = 0
        self.seam_warnings = 0
        self.seam_mismatches = 0
        self.chunks_processed = 0

    @property
    def converted_blocks(self):
        return self.block_count + self.fluid_count

    def record_block(self, fallback_used=False, fallback_key=None):
        self.block_count += 1
        if fallback_used:
            self.fallback_blocks += 1
            if fallback_key:
                self.fallback_keys[fallback_key] += 1

    def record_fluid(self):
        self.fluid_count += 1

    def record_unknown_legacy(self, block_id, block_data):
        self.unknown_legacy[f"{block_id}:{block_data}"] += 1

    def record_unknown_modern(self, name):
        self.unknown_modern[name] += 1

    def merge(self, other):
        self.block_count += other.block_count
        self.fluid_count += other.fluid_count
        self.fallback_blocks += other.fallback_blocks
        self.bounds_errors += other.bounds_errors
        self.hytale_oob += other.hytale_oob
        self.y_oob += other.y_oob
        self.seam_warnings += other.seam_warnings
        self.seam_mismatches += other.seam_mismatches
        self.chunks_processed += other.chunks_processed
        for key, value in other.fallback_keys.items():
            self.fallback_keys[key] += value
        for key, value in other.unknown_legacy.items():
            self.unknown_legacy[key] += value
        for key, value in other.unknown_modern.items():
            self.unknown_modern[key] += value


def stats_to_payload(stats):
    return {
        "block_count": stats.block_count,
        "fluid_count": stats.fluid_count,
        "fallback_blocks": stats.fallback_blocks,
        "fallback_keys": dict(stats.fallback_keys),
        "unknown_legacy": dict(stats.unknown_legacy),
        "unknown_modern": dict(stats.unknown_modern),
        "bounds_errors": stats.bounds_errors,
        "hytale_oob": stats.hytale_oob,
        "y_oob": stats.y_oob,
        "seam_warnings": stats.seam_warnings,
        "seam_mismatches": stats.seam_mismatches,
        "chunks_processed": stats.chunks_processed,
    }


def stats_from_payload(payload):
    stats = ConversionStats()
    if not payload:
        return stats
    stats.block_count = payload.get("block_count", 0)
    stats.fluid_count = payload.get("fluid_count", 0)
    stats.fallback_blocks = payload.get("fallback_blocks", 0)
    stats.bounds_errors = payload.get("bounds_errors", 0)
    stats.hytale_oob = payload.get("hytale_oob", 0)
    stats.y_oob = payload.get("y_oob", 0)
    stats.seam_warnings = payload.get("seam_warnings", 0)
    stats.seam_mismatches = payload.get("seam_mismatches", 0)
    stats.chunks_processed = payload.get("chunks_processed", 0)
    for key, value in payload.get("fallback_keys", {}).items():
        stats.fallback_keys[key] += value
    for key, value in payload.get("unknown_legacy", {}).items():
        stats.unknown_legacy[key] += value
    for key, value in payload.get("unknown_modern", {}).items():
        stats.unknown_modern[key] += value
    return stats


class ChunkDebug:
    def __init__(self, chunk_x, chunk_z, y_offset, fallback_material):
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z
        self.origin_x = chunk_x * 16
        self.origin_z = chunk_z * 16
        self.y_offset = y_offset
        self.fallback_material = fallback_material
        self.min_y = None
        self.max_y = None
        self.block_count = 0
        self.fluid_count = 0
        self.fallback_blocks = 0
        self.material_counts = Counter()
        self.heightmap = [[None] * 16 for _ in range(16)]
        self.heightmap_material = [[None] * 16 for _ in range(16)]
        self.surface_y = [[None] * 16 for _ in range(16)]
        self.surface_material = [[None] * 16 for _ in range(16)]

    def record_block(self, x, y, z, material, fallback_used=False):
        self.block_count += 1
        self.material_counts[material] += 1
        if fallback_used:
            self.fallback_blocks += 1
        if self.min_y is None or y < self.min_y:
            self.min_y = y
        if self.max_y is None or y > self.max_y:
            self.max_y = y
        current = self.heightmap[z][x]
        if current is None or y > current:
            self.heightmap[z][x] = y
            self.heightmap_material[z][x] = material
        surface = self.surface_y[z][x]
        if surface is None or y >= surface:
            self.surface_y[z][x] = y
            self.surface_material[z][x] = material

    def record_fluid(self, x, y, z, fluid_name):
        self.fluid_count += 1
        material = f"Fluid_{fluid_name}"
        surface = self.surface_y[z][x]
        if surface is None or y >= surface:
            self.surface_y[z][x] = y
            self.surface_material[z][x] = material

    def to_report(self):
        top_materials = [
            {"material": name, "count": count}
            for name, count in self.material_counts.most_common(10)
        ]
        fallback_percent = 0.0
        if self.block_count:
            fallback_percent = (self.fallback_blocks / self.block_count) * 100.0
        return {
            "mc_chunk_x": self.chunk_x,
            "mc_chunk_z": self.chunk_z,
            "computed_world_origin_xz": [self.origin_x, self.origin_z],
            "y_offset": self.y_offset,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "blocks_output": self.block_count,
            "fluids_output": self.fluid_count,
            "top_materials": top_materials,
            "fallback_percent": fallback_percent,
        }

    def edge_heightmap(self, direction):
        if direction == "west":
            return [
                (self.heightmap[z][0], self.heightmap_material[z][0])
                for z in range(16)
            ]
        if direction == "east":
            return [
                (self.heightmap[z][15], self.heightmap_material[z][15])
                for z in range(16)
            ]
        if direction == "north":
            return [
                (self.heightmap[0][x], self.heightmap_material[0][x])
                for x in range(16)
            ]
        if direction == "south":
            return [
                (self.heightmap[15][x], self.heightmap_material[15][x])
                for x in range(16)
            ]
        return []


class SeamTracker:
    def __init__(self, mismatch_threshold=SEAM_MISMATCH_THRESHOLD):
        self.mismatch_threshold = mismatch_threshold
        self.pending = {}
        self.warning_count = 0
        self.mismatch_count = 0

    def _edge_mismatches(self, edge_a, edge_b):
        mismatches = 0
        for (height_a, _material_a), (height_b, _material_b) in zip(edge_a, edge_b):
            if height_a is None and height_b is None:
                continue
            if height_a is None or height_b is None:
                mismatches += 1
                continue
            if abs(height_a - height_b) > 1:
                mismatches += 1
        return mismatches

    def _check_neighbor(self, chunk_x, chunk_z, direction, edge):
        neighbor = self.pending.get((chunk_x, chunk_z))
        if not neighbor:
            return 0
        neighbor_edge = neighbor.get(direction)
        if not neighbor_edge:
            return 0
        mismatches = self._edge_mismatches(edge, neighbor_edge)
        neighbor[direction] = None
        if neighbor.get("east") is None and neighbor.get("south") is None:
            self.pending.pop((chunk_x, chunk_z), None)
        return mismatches

    def check_and_store(self, chunk_debug):
        chunk_x = chunk_debug.chunk_x
        chunk_z = chunk_debug.chunk_z
        west_edge = chunk_debug.edge_heightmap("west")
        north_edge = chunk_debug.edge_heightmap("north")
        mismatches = 0
        mismatches += self._check_neighbor(chunk_x - 1, chunk_z, "east", west_edge)
        mismatches += self._check_neighbor(chunk_x, chunk_z - 1, "south", north_edge)
        if mismatches > self.mismatch_threshold:
            self.warning_count += 1
            self.mismatch_count += mismatches
            print(
                "Seam warning: chunk "
                f"({chunk_x}, {chunk_z}) had {mismatches} border mismatches.",
                flush=True,
            )
        self.pending[(chunk_x, chunk_z)] = {
            "east": chunk_debug.edge_heightmap("east"),
            "south": chunk_debug.edge_heightmap("south"),
        }


class DebugExporter:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.chunks_dir = os.path.join(root_dir, "chunks")
        _safe_mkdir(self.chunks_dir)
        self.image_lib = _try_import_pil()
        self.image_ext = ".png" if self.image_lib else ""

    def export_chunk(self, chunk_debug):
        report_path = os.path.join(
            self.chunks_dir,
            f"chunk_{chunk_debug.chunk_x}_{chunk_debug.chunk_z}.json",
        )
        atomic_write_json(report_path, chunk_debug.to_report())
        height_path = os.path.join(
            self.chunks_dir,
            f"chunk_{chunk_debug.chunk_x}_{chunk_debug.chunk_z}_height"
            f"{self.image_ext or '.pgm'}",
        )
        surface_path = os.path.join(
            self.chunks_dir,
            f"chunk_{chunk_debug.chunk_x}_{chunk_debug.chunk_z}_surface"
            f"{self.image_ext or '.ppm'}",
        )
        _write_heightmap_image(height_path, chunk_debug.heightmap, self.image_lib)
        _write_surface_image(surface_path, chunk_debug.surface_material, self.image_lib)


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


def build_chunk_blob(builder, template_info, block_chunk_data=None):
    sections_out = builder.build_sections(
        template_info.asset_count,
        template_info.empty_fluid_data,
        template_info.block_physics_data,
    )
    if block_chunk_data is None:
        block_chunk_data = template_info.block_chunk_data
    chunk_doc = {
        "Components": {
            "BlockComponentChunk": {"BlockComponents": {}},
            "EnvironmentChunk": {"Data": Binary(template_info.environment_data)},
            "ChunkColumn": {"Sections": sections_out},
            "WorldChunk": {},
            "BlockHealthChunk": {"Data": Binary(template_info.block_health_data)},
            "BlockChunk": {
                "Version": template_info.block_chunk_version,
                "Data": Binary(block_chunk_data),
            },
            "EntityChunk": {"Entities": []},
        }
    }
    return BSON.encode(chunk_doc)

def process_mc_chunk(
    chunk_x,
    chunk_z,
    sections,
    mapper,
    template_info,
    chunks,
    stats,
    y_offset=0,
    debug_exporter=None,
    seam_tracker=None,
    validate=False,
):
    chunk_debug = None
    if debug_exporter or seam_tracker or validate:
        chunk_debug = ChunkDebug(chunk_x, chunk_z, y_offset, mapper.default)
    max_y = template_info.section_count * 32 - 1

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

                x = idx & 0x0F
                z = (idx >> 4) & 0x0F
                y = (idx >> 8) & 0x0F

                if validate and (x < 0 or x > 15 or z < 0 or z > 15):
                    stats.bounds_errors += 1

                global_x = chunk_x * 16 + x
                global_z = chunk_z * 16 + z
                global_y = sec_y * 16 + y + y_offset

                if global_y < 0 or global_y > max_y:
                    stats.y_oob += 1
                    continue

                fluid = fluid_from_legacy(block_id, block_data)
                if fluid is not None:
                    fluid_name, fluid_level = fluid
                    builder, h_local_x, h_local_z = get_hytale_builder(
                        chunks, template_info, global_x, global_z
                    )
                    if validate and (
                        h_local_x < 0
                        or h_local_x > 31
                        or h_local_z < 0
                        or h_local_z > 31
                    ):
                        stats.hytale_oob += 1
                        continue
                    builder.set_fluid(
                        h_local_x, global_y, h_local_z, fluid_name, fluid_level
                    )
                    stats.record_fluid()
                    if chunk_debug:
                        chunk_debug.record_fluid(x, global_y, z, fluid_name)
                    continue

                block_name, map_source, fallback_key = mapper.map_legacy_info(
                    block_id, block_data
                )
                if map_source == "default":
                    stats.record_unknown_legacy(block_id, block_data)
                fluid_override = parse_fluid_mapping(block_name)
                if fluid_override is not None:
                    fluid_name, fluid_level = fluid_override
                    builder, h_local_x, h_local_z = get_hytale_builder(
                        chunks, template_info, global_x, global_z
                    )
                    if validate and (
                        h_local_x < 0
                        or h_local_x > 31
                        or h_local_z < 0
                        or h_local_z > 31
                    ):
                        stats.hytale_oob += 1
                        continue
                    builder.set_fluid(
                        h_local_x, global_y, h_local_z, fluid_name, fluid_level
                    )
                    stats.record_fluid()
                    if chunk_debug:
                        chunk_debug.record_fluid(x, global_y, z, fluid_name)
                    continue
                if block_name == "Empty":
                    continue

                builder, h_local_x, h_local_z = get_hytale_builder(
                    chunks, template_info, global_x, global_z
                )
                if validate and (
                    h_local_x < 0
                    or h_local_x > 31
                    or h_local_z < 0
                    or h_local_z > 31
                ):
                    stats.hytale_oob += 1
                    continue
                fallback_used = map_source != "direct"
                stats.record_block(fallback_used, fallback_key)
                builder.set_block(h_local_x, global_y, h_local_z, block_name)
                if chunk_debug:
                    chunk_debug.record_block(
                        x, global_y, z, block_name, fallback_used=fallback_used
                    )
        elif palette is not None:
            palette_list = list(palette)
            if not palette_list:
                continue
            if block_states is None:
                entry = palette_list[0]
                name, properties = entry_name_and_properties(entry)
                if is_air_name(name):
                    continue
                base_global_x = chunk_x * 16
                base_global_z = chunk_z * 16
                h_chunk_x, base_local_x = floor_divmod(base_global_x, 32)
                h_chunk_z, base_local_z = floor_divmod(base_global_z, 32)
                chunk_key = (h_chunk_x, h_chunk_z)
                builder = chunks.get(chunk_key)
                if builder is None:
                    builder = HytaleChunkBuilder(template_info.section_count)
                    chunks[chunk_key] = builder
                fluid = fluid_from_modern(name, properties)
                state_key = build_state_key(entry)
                block_name, map_source, fallback_key = mapper.map_modern_info(
                    state_key
                )
                fluid_override = parse_fluid_mapping(block_name)
                for local_y in range(16):
                    global_y = sec_y * 16 + local_y + y_offset
                    if global_y < 0 or global_y > max_y:
                        stats.y_oob += 1
                        continue
                    for z in range(16):
                        h_local_z = base_local_z + z
                        for x in range(16):
                            h_local_x = base_local_x + x
                            if validate and (
                                h_local_x < 0
                                or h_local_x > 31
                                or h_local_z < 0
                                or h_local_z > 31
                            ):
                                stats.hytale_oob += 1
                                continue
                            if fluid is not None:
                                fluid_name, fluid_level = fluid
                                builder.set_fluid(
                                    h_local_x,
                                    global_y,
                                    h_local_z,
                                    fluid_name,
                                    fluid_level,
                                )
                                stats.record_fluid()
                                if chunk_debug:
                                    chunk_debug.record_fluid(
                                        x, global_y, z, fluid_name
                                    )
                                continue
                            if fluid_override is not None:
                                fluid_name, fluid_level = fluid_override
                                builder.set_fluid(
                                    h_local_x,
                                    global_y,
                                    h_local_z,
                                    fluid_name,
                                    fluid_level,
                                )
                                stats.record_fluid()
                                if chunk_debug:
                                    chunk_debug.record_fluid(
                                        x, global_y, z, fluid_name
                                    )
                                continue
                            if block_name == "Empty":
                                continue
                            fallback_used = map_source != "direct"
                            stats.record_block(fallback_used, fallback_key)
                            if map_source == "default":
                                stats.record_unknown_modern(
                                    normalize_block_name(state_key)
                                )
                            builder.set_block(
                                h_local_x, global_y, h_local_z, block_name
                            )
                            if chunk_debug:
                                chunk_debug.record_block(
                                    x,
                                    global_y,
                                    z,
                                    block_name,
                                    fallback_used=fallback_used,
                                )
                continue

            indices = decode_block_states(block_states, len(palette_list))
            for idx, palette_index in enumerate(indices):
                if palette_index >= len(palette_list):
                    continue
                entry = palette_list[palette_index]
                name, properties = entry_name_and_properties(entry)
                if is_air_name(name):
                    continue

                x = idx & 0x0F
                z = (idx >> 4) & 0x0F
                y = (idx >> 8) & 0x0F

                if validate and (x < 0 or x > 15 or z < 0 or z > 15):
                    stats.bounds_errors += 1

                global_x = chunk_x * 16 + x
                global_z = chunk_z * 16 + z
                global_y = sec_y * 16 + y + y_offset

                if global_y < 0 or global_y > max_y:
                    stats.y_oob += 1
                    continue

                fluid = fluid_from_modern(name, properties)
                if fluid is not None:
                    fluid_name, fluid_level = fluid
                    builder, h_local_x, h_local_z = get_hytale_builder(
                        chunks, template_info, global_x, global_z
                    )
                    if validate and (
                        h_local_x < 0
                        or h_local_x > 31
                        or h_local_z < 0
                        or h_local_z > 31
                    ):
                        stats.hytale_oob += 1
                        continue
                    builder.set_fluid(
                        h_local_x, global_y, h_local_z, fluid_name, fluid_level
                    )
                    stats.record_fluid()
                    if chunk_debug:
                        chunk_debug.record_fluid(x, global_y, z, fluid_name)
                    continue

                state_key = build_state_key(entry)
                block_name, map_source, fallback_key = mapper.map_modern_info(
                    state_key
                )
                if map_source == "default":
                    stats.record_unknown_modern(normalize_block_name(state_key))
                fluid_override = parse_fluid_mapping(block_name)
                if fluid_override is not None:
                    fluid_name, fluid_level = fluid_override
                    builder, h_local_x, h_local_z = get_hytale_builder(
                        chunks, template_info, global_x, global_z
                    )
                    if validate and (
                        h_local_x < 0
                        or h_local_x > 31
                        or h_local_z < 0
                        or h_local_z > 31
                    ):
                        stats.hytale_oob += 1
                        continue
                    builder.set_fluid(
                        h_local_x, global_y, h_local_z, fluid_name, fluid_level
                    )
                    stats.record_fluid()
                    if chunk_debug:
                        chunk_debug.record_fluid(x, global_y, z, fluid_name)
                    continue
                if block_name == "Empty":
                    continue

                builder, h_local_x, h_local_z = get_hytale_builder(
                    chunks, template_info, global_x, global_z
                )
                if validate and (
                    h_local_x < 0
                    or h_local_x > 31
                    or h_local_z < 0
                    or h_local_z > 31
                ):
                    stats.hytale_oob += 1
                    continue
                fallback_used = map_source != "direct"
                stats.record_block(fallback_used, fallback_key)
                builder.set_block(h_local_x, global_y, h_local_z, block_name)
                if chunk_debug:
                    chunk_debug.record_block(
                        x, global_y, z, block_name, fallback_used=fallback_used
                    )

    if chunk_debug:
        if seam_tracker:
            seam_tracker.check_and_store(chunk_debug)
        if debug_exporter:
            debug_exporter.export_chunk(chunk_debug)
    stats.chunks_processed += 1


def build_region_chunk_blobs(
    region_path,
    mapper,
    template_info,
    y_offset=0,
    debug_exporter=None,
    seam_tracker=None,
    validate=False,
):
    chunks = {}
    stats = ConversionStats()
    seam_warnings_before = seam_tracker.warning_count if seam_tracker else 0
    seam_mismatches_before = seam_tracker.mismatch_count if seam_tracker else 0

    for chunk_x, chunk_z, sections in iter_mc_chunks(region_path):
        process_mc_chunk(
            chunk_x,
            chunk_z,
            sections,
            mapper,
            template_info,
            chunks,
            stats,
            y_offset=y_offset,
            debug_exporter=debug_exporter,
            seam_tracker=seam_tracker,
            validate=validate,
        )

    if seam_tracker:
        stats.seam_warnings += seam_tracker.warning_count - seam_warnings_before
        stats.seam_mismatches += seam_tracker.mismatch_count - seam_mismatches_before

    entries = []
    for (chunk_x, chunk_z), builder in chunks.items():
        block_chunk_data = build_block_chunk_data(builder, template_info)
        blob = build_chunk_blob(builder, template_info, block_chunk_data=block_chunk_data)
        entries.append((chunk_x, chunk_z, blob))
    return entries, stats, len(entries) == 0


_WORKER_TEMPLATE_INFO = None
_WORKER_MAPPER = None
_WORKER_DEBUG_EXPORTER = None
_WORKER_SEAM_TRACKER = None
_WORKER_Y_OFFSET = 0
_WORKER_VALIDATE = False


def _init_region_worker(
    template_cache, mapping_path, default_block, y_offset, debug_export_dir, validate
):
    global _WORKER_TEMPLATE_INFO
    global _WORKER_MAPPER
    global _WORKER_DEBUG_EXPORTER
    global _WORKER_SEAM_TRACKER
    global _WORKER_Y_OFFSET
    global _WORKER_VALIDATE
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    _WORKER_TEMPLATE_INFO = TemplateInfo.from_cache(template_cache)
    _WORKER_MAPPER = BlockMapper(mapping_path, default_block=default_block)
    _WORKER_Y_OFFSET = y_offset
    _WORKER_VALIDATE = validate
    if debug_export_dir:
        _WORKER_DEBUG_EXPORTER = DebugExporter(debug_export_dir)
    else:
        _WORKER_DEBUG_EXPORTER = None
    _WORKER_SEAM_TRACKER = SeamTracker() if validate else None


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
    entries, stats, is_empty = build_region_chunk_blobs(
        region_path,
        _WORKER_MAPPER,
        _WORKER_TEMPLATE_INFO,
        y_offset=_WORKER_Y_OFFSET,
        debug_exporter=_WORKER_DEBUG_EXPORTER,
        seam_tracker=_WORKER_SEAM_TRACKER,
        validate=_WORKER_VALIDATE,
    )
    write_chunk_cache(cache_path, entries)
    return stats, is_empty


def index_block(x, y, z):
    return ((y & 31) << 10) | ((z & 31) << 5) | (x & 31)


def index_column(x, z):
    return ((z & 31) << 5) | (x & 31)


def fluid_max_level(name):
    if name in FLUID_MAX_LEVELS:
        return FLUID_MAX_LEVELS[name]
    if name.endswith("_Source"):
        return 1
    return FLUID_MAX_LEVEL_DEFAULT


def normalize_fluid_name(name, level):
    if name in FLUID_LEGACY_NAME_MAP:
        normal, source = FLUID_LEGACY_NAME_MAP[name]
        if source and level == 0:
            return source
        return normal
    if name in FLUID_MAX_LEVELS:
        return name
    if name.startswith("Fluid_"):
        return name.split("Fluid_", 1)[1]
    return None


def parse_fluid_mapping(block_name):
    if not block_name:
        return None
    base = block_name.split("|", 1)[0]
    level = None
    if "|FluidLevel=" in block_name:
        parts = block_name.split("|")
        for part in parts[1:]:
            if part.startswith("FluidLevel="):
                try:
                    level = int(part.split("=", 1)[1])
                except ValueError:
                    level = None
                break
    level = 0 if level is None else level
    fluid_name = normalize_fluid_name(base, level)
    if fluid_name is None:
        return None
    if level <= 0:
        level = fluid_max_level(fluid_name)
    level = max(0, min(15, level))
    return fluid_name, level


def mc_fluid_to_hytale(base, level, falling=False):
    base = base.lower()
    is_water = base == "water"
    max_level = FLUID_MAX_LEVEL_DEFAULT
    if level < 0:
        level = 0
    if level == 0 and not falling:
        name = "Water_Source" if is_water else "Lava_Source"
        return name, fluid_max_level(name)
    name = "Water" if is_water else "Lava"
    if falling or level >= max_level:
        return name, max_level
    return name, max(1, max_level - level)


def fluid_from_legacy(block_id, block_data):
    if block_id in (8, 9):
        base = "water"
        is_source = block_id == 9
    elif block_id in (10, 11):
        base = "lava"
        is_source = block_id == 11
    else:
        return None
    level = block_data & 0x7
    falling = (block_data & 0x8) != 0
    if is_source and level == 0 and not falling:
        name = "Water_Source" if base == "water" else "Lava_Source"
        return name, fluid_max_level(name)
    return mc_fluid_to_hytale(base, level, falling)


def fluid_from_modern(name, properties):
    if not name:
        return None
    base = name.split(":", 1)[1] if name.startswith("minecraft:") else name
    if base not in ("water", "flowing_water", "lava", "flowing_lava"):
        return None
    is_flowing = base.startswith("flowing_")
    base = base.replace("flowing_", "")
    level = 0
    if properties:
        raw_level = properties.get("level")
        if raw_level is not None:
            try:
                level = int(raw_level)
            except ValueError:
                level = 0
    falling = False
    if properties:
        raw_falling = properties.get("falling")
        if isinstance(raw_falling, str):
            falling = raw_falling.lower() == "true"
        elif raw_falling is True:
            falling = True
    if is_flowing and level == 0:
        level = 1
    base = "water" if "water" in base else "lava"
    return mc_fluid_to_hytale(base, level, falling)


def is_air_name(name):
    if not name:
        return True
    base = name.split(":", 1)[1] if name.startswith("minecraft:") else name
    return base in ("air", "cave_air", "void_air")


def entry_name_and_properties(entry):
    name = entry.get("Name") or entry.get("name", "")
    properties = entry.get("Properties")
    if properties is None:
        properties = entry.get("properties") or {}
    return name, properties


def get_hytale_builder(chunks, template_info, global_x, global_z):
    h_chunk_x, h_local_x = floor_divmod(global_x, 32)
    h_chunk_z, h_local_z = floor_divmod(global_z, 32)
    chunk_key = (h_chunk_x, h_chunk_z)
    builder = chunks.get(chunk_key)
    if builder is None:
        builder = HytaleChunkBuilder(template_info.section_count)
        chunks[chunk_key] = builder
    return builder, h_local_x, h_local_z


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


def build_empty_fluid_data():
    return bytes([PALETTE_EMPTY, 0])


def build_fluid_palette(fluids):
    if not fluids:
        return build_empty_fluid_data()

    unique = {name for name, _level in fluids.values()}
    unique.add("Empty")

    if len(unique) <= 16:
        palette_type = PALETTE_HALF_BYTE
        blocks_len = HALF_BYTE_BLOCKS_LEN
    elif len(unique) <= 256:
        palette_type = PALETTE_BYTE
        blocks_len = BYTE_BLOCKS_LEN
    else:
        raise ValueError(f"Too many unique fluids in section: {len(unique)}")

    names = ["Empty"] + sorted(name for name in unique if name != "Empty")
    internal_by_name = {name: idx for idx, name in enumerate(names)}

    if palette_type == PALETTE_HALF_BYTE:
        block_bytes = bytearray([0]) * blocks_len
    else:
        block_bytes = bytearray([0]) * blocks_len

    level_data = bytearray(HALF_BYTE_BLOCKS_LEN)
    counts = defaultdict(int)

    for idx, (name, level) in fluids.items():
        if name == "Empty" or not level:
            continue
        internal_id = internal_by_name.get(name)
        if internal_id is None:
            continue
        counts[internal_id] += 1
        if palette_type == PALETTE_HALF_BYTE:
            set_nibble(block_bytes, idx, internal_id)
        else:
            block_bytes[idx] = internal_id & 0xFF
        set_nibble(level_data, idx, level)

    empty_id = internal_by_name["Empty"]
    counts[empty_id] = BLOCKS_PER_SECTION - sum(counts.values())

    buf = bytearray()
    buf.append(palette_type)
    buf.extend(struct.pack(">H", len(names)))
    for internal_id, name in enumerate(names):
        buf.append(internal_id & 0xFF)
        write_utf(buf, name)
        buf.extend(struct.pack(">H", counts.get(internal_id, 0)))

    buf.extend(block_bytes)
    buf.append(1)
    buf.extend(level_data)
    return bytes(buf)


def _build_bitfield(values, bits, length):
    total_bits = length * bits
    buf = bytearray(total_bits // 8)
    for index, value in enumerate(values):
        bit_index = index * bits
        for bit in range(bits):
            if value & (1 << bit):
                byte_index = (bit_index + bit) >> 3
                buf[byte_index] |= 1 << ((bit_index + bit) & 7)
    return bytes(buf)


def _parse_bitfield(data, bits, length):
    values = []
    for index in range(length):
        bit_index = index * bits
        value = 0
        for bit in range(bits):
            byte_index = (bit_index + bit) >> 3
            if data[byte_index] & (1 << ((bit_index + bit) & 7)):
                value |= 1 << bit
        values.append(value)
    return values


def parse_short_palette(data, offset, length=1024, bits=10):
    count = struct.unpack_from("<h", data, offset)[0]
    offset += 2
    if count <= 0:
        count = 1
    keys = list(struct.unpack_from("<" + "h" * count, data, offset))
    offset += 2 * count
    field_len = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    field = data[offset : offset + field_len]
    offset += field_len
    ids = _parse_bitfield(field, bits, length)
    values = [keys[idx] if idx < len(keys) else 0 for idx in ids]
    return values, offset


def parse_int_palette(data, offset, length=1024, bits=10):
    count = struct.unpack_from("<h", data, offset)[0]
    offset += 2
    if count <= 0:
        count = 1
    keys = list(struct.unpack_from("<" + "i" * count, data, offset))
    offset += 4 * count
    field_len = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    field = data[offset : offset + field_len]
    offset += field_len
    ids = _parse_bitfield(field, bits, length)
    values = [keys[idx] if idx < len(keys) else 0 for idx in ids]
    return values, offset


def extract_default_tint(block_chunk_data):
    if not block_chunk_data:
        return 0
    try:
        offset = 1
        _heights, offset = parse_short_palette(block_chunk_data, offset)
        tints, _ = parse_int_palette(block_chunk_data, offset)
    except Exception:
        return 0
    if not tints:
        return 0
    return Counter(tints).most_common(1)[0][0]


def build_short_palette(values):
    keys = []
    key_index = {}
    ids = []
    for value in values:
        key = int(value)
        idx = key_index.get(key)
        if idx is None:
            idx = len(keys)
            key_index[key] = idx
            keys.append(key)
        ids.append(idx)
    count = len(keys)
    if count <= 0:
        count = 1
        keys = [0]
        ids = [0] * len(values)
    buf = bytearray()
    buf.extend(struct.pack("<h", count))
    for key in keys:
        buf.extend(struct.pack("<h", key))
    bitfield = _build_bitfield(ids, 10, len(values))
    buf.extend(struct.pack("<i", len(bitfield)))
    buf.extend(bitfield)
    return bytes(buf)


def build_int_palette(values):
    keys = []
    key_index = {}
    ids = []
    for value in values:
        key = int(value)
        idx = key_index.get(key)
        if idx is None:
            idx = len(keys)
            key_index[key] = idx
            keys.append(key)
        ids.append(idx)
    count = len(keys)
    if count <= 0:
        count = 1
        keys = [0]
        ids = [0] * len(values)
    buf = bytearray()
    buf.extend(struct.pack("<h", count))
    for key in keys:
        buf.extend(struct.pack("<i", key))
    bitfield = _build_bitfield(ids, 10, len(values))
    buf.extend(struct.pack("<i", len(bitfield)))
    buf.extend(bitfield)
    return bytes(buf)


def build_block_chunk_data(builder, template_info):
    needs_physics = 1
    if template_info.block_chunk_data:
        needs_physics = 1 if template_info.block_chunk_data[0] else 0
    heights = [min(319, max(0, h)) for h in builder.build_heightmap()]
    height_bytes = build_short_palette(heights)
    default_tint = template_info.default_tint
    if default_tint is None:
        default_tint = 0
    tint_values = [default_tint] * (32 * 32)
    tint_bytes = build_int_palette(tint_values)
    buf = bytearray()
    buf.append(needs_physics)
    buf.extend(height_bytes)
    buf.extend(tint_bytes)
    return bytes(buf)


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


FALLBACK_EXACT = {
    "grass_block": "Soil_Grass",
    "dirt": "Soil_Dirt",
    "coarse_dirt": "Soil_Dirt_Dry",
    "podzol": "Soil_Dirt_Dry",
    "mycelium": "Soil_Dirt_Poisoned",
    "rooted_dirt": "Soil_Dirt",
    "mud": "Soil_Dirt",
    "clay": "Soil_Dirt",
    "stone": "Rock_Stone",
    "cobblestone": "Rock_Stone",
    "gravel": "Soil_Gravel",
    "sand": "Soil_Gravel_Sand",
    "red_sand": "Soil_Gravel_Sand_Red",
    "sandstone": "Rock_Sandstone",
    "red_sandstone": "Rock_Sandstone_Red",
    "snow": "Soil_Snow_Half",
    "snow_block": "Soil_Snow",
    "powder_snow": "Soil_Snow",
    "ice": "Rock_Quartzite",
    "packed_ice": "Rock_Quartzite",
    "blue_ice": "Rock_Aqua",
    "bedrock": "Rock_Bedrock",
}

FALLBACK_CONTAINS = (
    ("grass", "Plant_Grass_Lush_Short"),
    ("fern", "Plant_Grass_Jungle_Short"),
    ("tall_grass", "Plant_Grass_Lush_Tall"),
    ("seagrass", "Plant_Seaweed_Grass"),
    ("kelp", "Plant_Seaweed_Grass_Tall"),
    ("andesite", "Rock_Shale"),
    ("diorite", "Rock_Calcite"),
    ("granite", "Rock_Marble"),
    ("deepslate", "Rock_Slate"),
    ("slate", "Rock_Slate"),
    ("basalt", "Rock_Basalt"),
    ("tuff", "Rock_Chalk"),
    ("dripstone", "Rock_Chalk"),
    ("calcite", "Rock_Calcite"),
    ("obsidian", "Rock_Volcanic"),
    ("sandstone", "Rock_Sandstone"),
    ("sand", "Soil_Gravel_Sand"),
    ("gravel", "Soil_Gravel"),
    ("stone", "Rock_Stone"),
)

WOOD_TYPE_MATERIAL = {
    "oak": "Wood_Oak_Trunk",
    "spruce": "Wood_Fir_Trunk",
    "birch": "Wood_Birch_Trunk",
    "jungle": "Wood_Jungle_Trunk",
    "acacia": "Wood_Dry_Trunk",
    "dark_oak": "Wood_Poisoned_Trunk",
    "mangrove": "Wood_Palm_Trunk",
    "cherry": "Wood_Maple_Trunk",
    "bamboo": "Wood_Bamboo_Trunk",
    "crimson": "Wood_Fire_Trunk",
    "warped": "Wood_Ice_Trunk",
}

LEAF_TYPE_MATERIAL = {
    "oak": "Plant_Leaves_Oak",
    "spruce": "Plant_Leaves_Fir",
    "birch": "Plant_Leaves_Birch",
    "jungle": "Plant_Leaves_Jungle",
    "acacia": "Plant_Leaves_Dry",
    "dark_oak": "Plant_Leaves_Poisoned",
    "mangrove": "Plant_Leaves_Palm",
    "cherry": "Plant_Leaves_Autumn",
    "azalea": "Plant_Leaves_Bramble",
}

WOOD_SUFFIXES = ("_log", "_wood", "_planks", "_stem", "_hyphae")
LEAF_SUFFIXES = ("_leaves",)

STRIP_SUFFIXES = (
    "_fence_gate",
    "_pressure_plate",
    "_trapdoor",
    "_button",
    "_stairs",
    "_slab",
    "_wall",
    "_fence",
    "_door",
    "_gate",
    "_sign",
    "_banner",
    "_carpet",
    "_wool",
    "_bed",
)


def normalize_block_name(name):
    if not name:
        return ""
    base = name.split("[", 1)[0]
    if base.startswith("minecraft:"):
        base = base.split(":", 1)[1]
    return base.lower()


def strip_variant_suffix(name):
    for suffix in STRIP_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def infer_tree_material(name, mapping, default_material):
    for key, material in mapping.items():
        if key in name:
            return material
    return default_material


def fallback_material_for_name(name):
    if not name:
        return None
    base = normalize_block_name(name)
    if base in FALLBACK_EXACT:
        return FALLBACK_EXACT[base]
    if base.endswith("_ore"):
        return "Rock_Stone"
    if base.endswith("_block"):
        block_base = base[: -len("_block")]
        if block_base in (
            "coal",
            "iron",
            "gold",
            "copper",
            "diamond",
            "lapis",
            "redstone",
            "emerald",
            "amethyst",
        ):
            return "Rock_Stone"
    if base == "flowering_azalea":
        return "Plant_Leaves_Bramble"
    if base == "cocoa":
        return "Plant_Leaves_Jungle"
    if "moss" in base:
        return "Soil_Grass"
    if "amethyst" in base:
        return "Rock_Quartzite"
    if "lichen" in base:
        return "Plant_Leaves_Oak"
    if "vine" in base:
        return "Plant_Leaves_Oak"
    if "mushroom" in base:
        return "Plant_Leaves_Oak"
    if "dripleaf" in base:
        return "Plant_Leaves_Jungle"
    for suffix in LEAF_SUFFIXES:
        if base.endswith(suffix):
            return infer_tree_material(base, LEAF_TYPE_MATERIAL, "Plant_Leaves_Oak")
    for suffix in WOOD_SUFFIXES:
        if base.endswith(suffix):
            return infer_tree_material(base, WOOD_TYPE_MATERIAL, "Wood_Oak_Trunk")
    if base in WOOD_TYPE_MATERIAL:
        return WOOD_TYPE_MATERIAL[base]
    if base in LEAF_TYPE_MATERIAL:
        return LEAF_TYPE_MATERIAL[base]
    for key, material in FALLBACK_CONTAINS:
        if key in base:
            return material
    stripped = strip_variant_suffix(base)
    if stripped != base:
        return fallback_material_for_name(stripped)
    return None


class BlockMapper:
    def __init__(self, mapping_path=None, default_block=None):
        self.mapping = {}
        self.legacy = {}
        self.legacy_by_id = {}
        self.default = "Unknown"
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
        block_name, _source, _fallback_key = self.map_modern_info(name)
        return block_name

    def map_modern_info(self, name):
        if name in self.mapping:
            return self.mapping[name], "direct", None
        if name.startswith("minecraft:"):
            short = name.split(":", 1)[1]
            if short in self.mapping:
                return self.mapping[short], "direct", None
        base = name.split("[", 1)[0]
        if base != name:
            if base in self.mapping:
                return self.mapping[base], "direct", None
            if base.startswith("minecraft:"):
                short_base = base.split(":", 1)[1]
                if short_base in self.mapping:
                    return self.mapping[short_base], "direct", None
        fallback = fallback_material_for_name(name)
        if fallback:
            return fallback, "fallback", normalize_block_name(name)
        return self.default, "default", normalize_block_name(name)

    def map_legacy(self, block_id, block_data):
        block_name, _source, _fallback_key = self.map_legacy_info(block_id, block_data)
        return block_name

    def map_legacy_info(self, block_id, block_data):
        key = f"{block_id}:{block_data}"
        if key in self.legacy:
            return self.legacy[key], "direct", None
        if str(block_id) in self.legacy_by_id:
            return self.legacy_by_id[str(block_id)], "direct", None
        return self.default, "default", key

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
        self.default_tint = extract_default_tint(self.block_chunk_data)

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
        info.default_tint = payload.get("default_tint")
        if info.default_tint is None:
            info.default_tint = extract_default_tint(info.block_chunk_data)
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
            "default_tint": self.default_tint,
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
        self.fluids = [dict() for _ in range(section_count)]
        self.heightmap_solid = [-1] * (32 * 32)
        self.heightmap_any = [-1] * (32 * 32)

    def _update_height(self, x, y, z, solid=True):
        idx = index_column(x, z)
        if y > self.heightmap_any[idx]:
            self.heightmap_any[idx] = y
        if solid and y > self.heightmap_solid[idx]:
            self.heightmap_solid[idx] = y

    def set_block(self, x, y, z, block_name):
        section_index = y // 32
        if section_index < 0 or section_index >= self.section_count:
            return
        local_y = y % 32
        idx = index_block(x, local_y, z)
        if block_name != "Empty":
            self.sections[section_index][idx] = block_name
            self._update_height(x, y, z, solid=True)

    def set_fluid(self, x, y, z, fluid_name, level):
        section_index = y // 32
        if section_index < 0 or section_index >= self.section_count:
            return
        local_y = y % 32
        idx = index_block(x, local_y, z)
        if not fluid_name or fluid_name == "Empty":
            self.fluids[section_index].pop(idx, None)
            return
        if level <= 0:
            self.fluids[section_index].pop(idx, None)
            return
        self.fluids[section_index][idx] = (fluid_name, level & 0x0F)
        self._update_height(x, y, z, solid=False)

    def build_heightmap(self):
        heights = []
        for idx in range(32 * 32):
            height = self.heightmap_solid[idx]
            if height < 0:
                height = self.heightmap_any[idx]
            if height < 0:
                height = 0
            heights.append(height)
        return heights

    def build_sections(self, asset_count, empty_fluid_data, block_physics_data):
        sections_out = []
        empty_fluid = empty_fluid_data or build_empty_fluid_data()
        for section_blocks, section_fluids in zip(self.sections, self.fluids):
            block_data = build_palette(section_blocks, asset_count)
            if section_fluids:
                fluid_data = build_fluid_palette(section_fluids)
            else:
                fluid_data = empty_fluid
            components = {
                "ChunkSection": {},
                "Block": {"Version": 6, "Data": Binary(block_data)},
                "Fluid": {"Data": Binary(fluid_data)},
                "BlockPhysics": {"Data": Binary(block_physics_data)},
            }
            sections_out.append({"Components": components})
        return sections_out


def floor_divmod(value, divisor):
    q = math.floor(value / divisor)
    r = value - q * divisor
    return int(q), int(r)


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
    mc_world,
    output_world,
    template,
    mapping_path,
    template_cache=None,
    default_block=None,
    y_offset=0,
    debug_exporter=None,
    validate=False,
    fallback_threshold=DEFAULT_FALLBACK_THRESHOLD,
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
    stats = ConversionStats()
    seam_tracker = SeamTracker() if validate else None

    region_dir = os.path.join(mc_world, "region")
    for filename in os.listdir(region_dir):
        if not filename.endswith(".mca"):
            continue
        region_has_blocks = False
        region_path = os.path.join(region_dir, filename)
        for chunk_x, chunk_z, sections in iter_mc_chunks(region_path):
            before = stats.converted_blocks
            process_mc_chunk(
                chunk_x,
                chunk_z,
                sections,
                mapper,
                template_info,
                chunks,
                stats,
                y_offset=y_offset,
                debug_exporter=debug_exporter,
                seam_tracker=seam_tracker,
                validate=validate,
            )
            if stats.converted_blocks > before:
                region_has_blocks = True
        if region_has_blocks:
            print(f"Region converted: {filename}")
        else:
            print(f"Region converted (empty): {filename}")
    prepare_output_world(output_world, template)

    # build region blobs
    region_blobs = defaultdict(dict)
    for (chunk_x, chunk_z), builder in chunks.items():
        block_chunk_data = build_block_chunk_data(builder, template_info)
        blob = build_chunk_blob(builder, template_info, block_chunk_data=block_chunk_data)

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

    if seam_tracker:
        stats.seam_warnings += seam_tracker.warning_count
        stats.seam_mismatches += seam_tracker.mismatch_count

    print_conversion_summary(
        stats,
        chunk_count=len(chunks),
        fallback_threshold=fallback_threshold,
        validate=validate,
    )


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
    y_offset=0,
    debug_export=False,
    debug_export_dir=None,
    validate=False,
    fallback_threshold=DEFAULT_FALLBACK_THRESHOLD,
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
        "y_offset": y_offset,
        "debug_export": debug_export,
        "debug_export_dir": debug_export_dir,
        "validate": validate,
        "fallback_threshold": fallback_threshold,
        "region_files": region_files,
        "processed_regions": [],
        "phase": "cache",
        "converted_blocks": 0,
        "stats": stats_to_payload(ConversionStats()),
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
    y_offset = state.get("y_offset", 0)
    debug_export = state.get("debug_export", False)
    debug_export_dir = state.get("debug_export_dir")
    if debug_export and not debug_export_dir:
        debug_export_dir = os.path.join(output_world, "debug_exports")
    validate = state.get("validate", False)
    fallback_threshold = state.get(
        "fallback_threshold", DEFAULT_FALLBACK_THRESHOLD
    )

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

    stats_total = stats_from_payload(state.get("stats"))
    total_converted = stats_total.converted_blocks

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
                initargs=(
                    template_cache,
                    mapping_path,
                    default_block,
                    y_offset,
                    debug_export_dir if debug_export else None,
                    validate,
                ),
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
                    region_stats, is_empty = future.result()
                    stats_total.merge(region_stats)
                    total_converted = stats_total.converted_blocks
                    processed.add(region_filename)
                    state["processed_regions"] = sorted(processed)
                    state["converted_blocks"] = total_converted
                    state["stats"] = stats_to_payload(stats_total)
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
            seam_tracker = SeamTracker() if validate else None
            debug_exporter = DebugExporter(debug_export_dir) if debug_export else None
            for idx, region_filename in enumerate(remaining, start=1):
                region_start = time.monotonic()
                print(
                    "Processing region "
                    f"{idx}/{total_regions}: {region_filename}",
                    flush=True,
                )
                region_path = os.path.join(region_dir, region_filename)
                cache_path = cache_path_for_region(cache_dir, region_filename)
                entries, region_stats, is_empty = build_region_chunk_blobs(
                    region_path,
                    mapper,
                    template_info,
                    y_offset=y_offset,
                    debug_exporter=debug_exporter,
                    seam_tracker=seam_tracker,
                    validate=validate,
                )
                write_chunk_cache(cache_path, entries)
                stats_total.merge(region_stats)
                total_converted = stats_total.converted_blocks
                processed.add(region_filename)
                state["processed_regions"] = sorted(processed)
                state["converted_blocks"] = total_converted
                state["stats"] = stats_to_payload(stats_total)
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
    total_chunks = 0
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
    total_chunks = sum(expected_counts.values())

    print_conversion_summary(
        stats_total,
        chunk_count=total_chunks,
        fallback_threshold=fallback_threshold,
        validate=validate,
        cached=True,
    )

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


def print_conversion_summary(
    stats,
    chunk_count,
    fallback_threshold,
    validate=False,
    cached=False,
):
    if cached:
        print(f"Converted {stats.converted_blocks} blocks using cached conversion.")
    else:
        print(
            f"Converted {stats.converted_blocks} blocks into {chunk_count} chunks."
        )
    print(
        "Fallback material usage: "
        f"{(stats.fallback_blocks / stats.block_count * 100.0) if stats.block_count else 0.0:.2f}% "
        f"(threshold {fallback_threshold:.2f}%)."
    )
    if validate:
        if stats.block_count and (
            (stats.fallback_blocks / stats.block_count * 100.0)
            > fallback_threshold
        ):
            print("WARNING: fallback material usage exceeds threshold.")
        if stats.bounds_errors or stats.hytale_oob or stats.y_oob:
            print(
                "Bounds warnings: "
                f"mc_oob={stats.bounds_errors}, "
                f"hytale_oob={stats.hytale_oob}, "
                f"y_oob={stats.y_oob}."
            )
        if stats.seam_warnings:
            print(
                "Seam warnings: "
                f"{stats.seam_warnings} warnings, "
                f"{stats.seam_mismatches} mismatched border columns."
            )

    if stats.unknown_legacy:
        print("Unmapped legacy blocks (id:data -> count):")
        for key, count in sorted(
            stats.unknown_legacy.items(), key=lambda item: -item[1]
        )[:20]:
            print(f"  {key} -> {count}")
    if stats.unknown_modern:
        print("Unmapped modern blocks (name -> count):")
        for key, count in sorted(
            stats.unknown_modern.items(), key=lambda item: -item[1]
        )[:20]:
            print(f"  {key} -> {count}")
    if stats.fallback_keys:
        print("Fallback-mapped blocks (name -> count):")
        for key, count in sorted(
            stats.fallback_keys.items(), key=lambda item: -item[1]
        )[:20]:
            print(f"  {key} -> {count}")


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
    y_offset=None,
    debug_export=None,
    validate=False,
    fallback_threshold=None,
):
    if continue_mode:
        state = load_worldcache_state(cache_dir)
        if not state:
            raise FileNotFoundError(
                f"No worldcache.json found in {cache_dir} to continue."
            )
        if debug_export is not None:
            if debug_export is True:
                debug_dir = os.path.join(state["output"], "debug_exports")
            else:
                debug_dir = debug_export
            state["debug_export"] = True
            state["debug_export_dir"] = debug_dir
        if validate:
            state["validate"] = True
        if fallback_threshold is not None:
            state["fallback_threshold"] = fallback_threshold
        if y_offset is not None:
            state["y_offset"] = y_offset
        save_worldcache_state(cache_dir, state)
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

    if y_offset is None:
        min_section_y, max_section_y = scan_mc_y_range(mc_world)
        if min_section_y < 0:
            y_offset = -min_section_y * 16
        else:
            y_offset = 0
        mc_min_y = min_section_y * 16
        mc_max_y = max_section_y * 16 + 15
        print(
            "Detected MC Y range "
            f"{mc_min_y}..{mc_max_y}; applying y_offset={y_offset}.",
            flush=True,
        )

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
            y_offset=y_offset,
            debug_export=bool(debug_export),
            debug_export_dir=(
                os.path.join(output_world, "debug_exports")
                if debug_export is True
                else debug_export
            ),
            validate=validate,
            fallback_threshold=(
                fallback_threshold
                if fallback_threshold is not None
                else DEFAULT_FALLBACK_THRESHOLD
            ),
        )
        save_worldcache_state(cache_dir, state)
        convert_world_cached(state, cache_dir, ignore_prompt=ignore_prompt)
        return

    debug_exporter = None
    if debug_export:
        debug_dir = (
            os.path.join(output_world, "debug_exports")
            if debug_export is True
            else debug_export
        )
        debug_exporter = DebugExporter(debug_dir)

    convert_world(
        mc_world,
        output_world,
        template,
        mapping_path,
        template_cache,
        default_block=default_block,
        y_offset=y_offset or 0,
        debug_exporter=debug_exporter,
        validate=validate,
        fallback_threshold=(
            fallback_threshold if fallback_threshold is not None else DEFAULT_FALLBACK_THRESHOLD
        ),
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
    parser.add_argument(
        "--y-offset",
        type=int,
        default=None,
        help="Additive Y offset applied to all block placements.",
    )
    parser.add_argument(
        "--debug-export",
        nargs="?",
        const=True,
        default=None,
        help=(
            "Write per-chunk debug reports and images. "
            "Optionally supply an output directory."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable validation checks (bounds, seams, fallback threshold).",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=None,
        help="Fallback material percentage threshold (default: 5%).",
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
        y_offset=args.y_offset,
        debug_export=args.debug_export,
        validate=args.validate,
        fallback_threshold=args.fallback_threshold,
    )


if __name__ == "__main__":
    main()
