#!/usr/bin/env python3
import argparse
import io
import json
import os
import struct
import gzip
import zlib
from collections import defaultdict

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
        sections = root.get("Sections", [])
        yield chunk_x, chunk_z, sections


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
    def __init__(self, mapping_path=None):
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
    name = palette_entry.get("Name", "")
    properties = palette_entry.get("Properties")
    if not properties:
        return name
    parts = []
    for key in sorted(properties.keys()):
        parts.append(f"{key}={properties[key]}")
    return f"{name}[{','.join(parts)}]"


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


def convert_world(mc_world, output_world, template, mapping_path):
    mapper = BlockMapper(mapping_path)
    template_info = TemplateInfo(template)

    # build chunk builders
    chunks = {}
    unknown_counts = defaultdict(int)
    converted_blocks = 0

    region_dir = os.path.join(mc_world, "region")
    for filename in os.listdir(region_dir):
        if not filename.endswith(".mca"):
            continue
        region_path = os.path.join(region_dir, filename)
        for chunk_x, chunk_z, sections in iter_mc_chunks(region_path):
            for sec in sections:
                sec_y = int(sec.get("Y", 0))
                blocks = sec.get("Blocks")
                data = sec.get("Data")
                palette = sec.get("Palette")
                block_states = sec.get("BlockStates")

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

    # build output world structure
    os.makedirs(output_world, exist_ok=True)
    os.makedirs(os.path.join(output_world, "chunks"), exist_ok=True)
    # copy config and resources
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

    # build region blobs
    region_blobs = defaultdict(dict)
    for (chunk_x, chunk_z), builder in chunks.items():
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
        blob = BSON.encode(chunk_doc)

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


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Minecraft map to a Hytale world format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Minecraft world folder (contains region/)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output Hytale world folder",
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to template Hytale world folder (e.g., serverexample/universe/worlds/default)",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional mapping JSON to override block mappings",
    )

    args = parser.parse_args()
    mapping_path = args.mapping
    if mapping_path is None:
        default_mapping = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mappings", "default.json"
        )
        if os.path.exists(default_mapping):
            mapping_path = default_mapping
    convert_world(args.input, args.output, args.template, mapping_path)


if __name__ == "__main__":
    main()
