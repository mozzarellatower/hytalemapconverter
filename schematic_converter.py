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

BLOCKS_PER_SECTION = 16 * 16 * 16


def read_mc_chunk_bytes(region_path, chunk_index):
    with open(region_path, "rb") as f:
        f.seek(chunk_index * 4)
        entry = struct.unpack(">I", f.read(4))[0]
        sector_offset = entry >> 8
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


def decode_packed_block_states(block_states, palette_size, total_blocks):
    if palette_size <= 1:
        return [0] * total_blocks
    bits_per_block = max(2, (palette_size - 1).bit_length())
    mask = (1 << bits_per_block) - 1
    longs = [value & 0xFFFFFFFFFFFFFFFF for value in block_states]
    indices = []
    for i in range(total_blocks):
        bit_index = i * bits_per_block
        long_index = bit_index >> 6
        start_bit = bit_index & 63
        value = (longs[long_index] >> start_bit) & mask
        if start_bit + bits_per_block > 64:
            value |= (longs[long_index + 1] << (64 - start_bit)) & mask
        indices.append(value)
    return indices


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


def iter_schematic_blocks(nbt_root, mapper):
    width = int(nbt_root.get("Width", 0))
    height = int(nbt_root.get("Height", 0))
    length = int(nbt_root.get("Length", 0))
    if width <= 0 or height <= 0 or length <= 0:
        raise ValueError("Schematic is missing Width/Height/Length.")

    blocks_tag = nbt_root.get("Blocks")
    data_tag = nbt_root.get("Data")
    if blocks_tag is None or data_tag is None:
        raise ValueError(
            "Unsupported schematic format. Expected legacy Blocks/Data arrays."
        )

    blocks_bytes = bytes(blocks_tag)
    data_bytes = bytes(data_tag)
    add_tag = nbt_root.get("AddBlocks")
    add_bytes = bytes(add_tag) if add_tag is not None else None
    total = width * height * length
    if len(blocks_bytes) < total or len(data_bytes) < (total + 1) // 2:
        raise ValueError("Schematic block arrays do not match dimensions.")

    for idx in range(total):
        block_id = blocks_bytes[idx]
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

        block_name = mapper.map_legacy(block_id, block_data)
        if block_name == "Empty":
            continue

        x = idx % width
        z = (idx // width) % length
        y = idx // (width * length)
        yield x, y, z, block_name


def decode_varint_array(data, count):
    values = []
    value = 0
    shift = 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            values.append(value)
            if count is not None and len(values) >= count:
                return values
            value = 0
            shift = 0
        else:
            shift += 7
            if shift > 35:
                raise ValueError("VarInt sequence is too long.")
    if count is not None and len(values) < count:
        raise ValueError("Not enough VarInts for schematic block data.")
    return values


def iter_schem_blocks(nbt_root, mapper):
    width = int(nbt_root.get("Width", 0))
    height = int(nbt_root.get("Height", 0))
    length = int(nbt_root.get("Length", 0))
    if width <= 0 or height <= 0 or length <= 0:
        raise ValueError("Schem is missing Width/Height/Length.")

    palette_tag = nbt_root.get("Palette")
    block_data_tag = nbt_root.get("BlockData")
    if palette_tag is None or block_data_tag is None:
        raise ValueError("Schem is missing Palette/BlockData.")

    palette_max = int(nbt_root.get("PaletteMax", 0))
    if palette_max <= 0:
        palette_max = max(int(v) for v in palette_tag.values()) + 1

    palette = [""] * palette_max
    for name, idx in palette_tag.items():
        palette[int(idx)] = str(name)

    total = width * height * length
    indices = decode_varint_array(bytes(block_data_tag), total)
    if len(indices) != total:
        raise ValueError("Schem block data length does not match dimensions.")

    for idx, palette_index in enumerate(indices):
        if palette_index >= len(palette):
            continue
        state_key = palette[palette_index]
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

        x = idx % width
        z = (idx // width) % length
        y = idx // (width * length)
        yield x, y, z, block_name


def vec3_from_nbt(value):
    if hasattr(value, "get"):
        if all(k in value for k in ("x", "y", "z")):
            return int(value["x"]), int(value["y"]), int(value["z"])
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return int(value[0]), int(value[1]), int(value[2])
    return 0, 0, 0


def iter_litematic_blocks(nbt_root, mapper):
    regions = nbt_root.get("Regions")
    if regions is None:
        raise ValueError("Litematic is missing Regions.")

    for region_name in regions:
        region = regions[region_name]
        pos = vec3_from_nbt(region.get("Position"))
        size = vec3_from_nbt(region.get("Size"))
        size_x = abs(size[0])
        size_y = abs(size[1])
        size_z = abs(size[2])
        if size_x == 0 or size_y == 0 or size_z == 0:
            continue

        base_x = pos[0] + (size[0] + 1 if size[0] < 0 else 0)
        base_y = pos[1] + (size[1] + 1 if size[1] < 0 else 0)
        base_z = pos[2] + (size[2] + 1 if size[2] < 0 else 0)

        palette_list = list(region.get("BlockStatePalette", []))
        block_states = region.get("BlockStates")
        if block_states is None or not palette_list:
            continue

        total = size_x * size_y * size_z
        indices = decode_packed_block_states(block_states, len(palette_list), total)

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

            x = idx % size_x
            z = (idx // size_x) % size_z
            y = idx // (size_x * size_z)

            yield base_x + x, base_y + y, base_z + z, block_name


def convert_schematic_to_prefab(
    schematic_path, output_path, mapping_path=None, default_block=None
):
    mapper = BlockMapper(mapping_path, default_block=default_block)
    nbt = nbtlib.load(schematic_path)
    root = nbt["Schematic"] if "Schematic" in nbt else nbt

    blocks = None
    if root.get("Regions") is not None:
        blocks = list(iter_litematic_blocks(root, mapper))
    elif root.get("Palette") is not None and root.get("BlockData") is not None:
        blocks = list(iter_schem_blocks(root, mapper))
    else:
        blocks = list(iter_schematic_blocks(root, mapper))
    if not blocks:
        raise RuntimeError("No blocks found to convert.")

    min_x = min(block[0] for block in blocks)
    min_y = min(block[1] for block in blocks)
    min_z = min(block[2] for block in blocks)
    max_x = max(block[0] for block in blocks)
    max_y = max(block[1] for block in blocks)
    max_z = max(block[2] for block in blocks)

    prefab_blocks = []
    for x, y, z, name in blocks:
        prefab_blocks.append(
            {
                "x": x - min_x,
                "y": y - min_y,
                "z": z - min_z,
                "name": name,
            }
        )

    prefab = {
        "version": 8,
        "blockIdVersion": 8,
        "anchorX": 0,
        "anchorY": 0,
        "anchorZ": 0,
        "blocks": prefab_blocks,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prefab, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Wrote prefab with {len(prefab_blocks)} blocks to {output_path}.")
    print(
        f"Bounds: min=({min_x},{min_y},{min_z}) max=({max_x},{max_y},{max_z})"
    )


def convert_world_to_prefab(
    mc_world, output_path, mapping_path=None, mode="auto", default_block=None
):
    if mode not in ("auto", "legacy", "modern"):
        raise ValueError(f"Unsupported mode: {mode}")

    if os.path.isfile(mc_world):
        return convert_schematic_to_prefab(
            mc_world, output_path, mapping_path, default_block=default_block
        )

    mapper = BlockMapper(mapping_path, default_block=default_block)
    blocks = []
    min_x = min_y = min_z = None
    max_x = max_y = max_z = None
    unknown_counts = defaultdict(int)

    region_dir = os.path.join(mc_world, "region")
    for filename in os.listdir(region_dir):
        if not filename.endswith(".mca"):
            continue
        region_path = os.path.join(region_dir, filename)
        for chunk_x, chunk_z, sections in iter_mc_chunks(region_path):
            for sec in sections:
                sec_y = int(sec.get("Y", 0))
                blocks_tag = sec.get("Blocks")
                data_tag = sec.get("Data")
                palette = sec.get("Palette")
                block_states = sec.get("BlockStates")

                if (
                    mode in ("auto", "legacy")
                    and blocks_tag is not None
                    and data_tag is not None
                ):
                    add_tag = sec.get("Add")
                    blocks_bytes = bytes(blocks_tag)
                    data_bytes = bytes(data_tag)
                    add_bytes = bytes(add_tag) if add_tag is not None else None

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

                        blocks.append((global_x, global_y, global_z, block_name))
                elif (
                    mode in ("auto", "modern")
                    and palette is not None
                    and block_states is not None
                ):
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

                        blocks.append((global_x, global_y, global_z, block_name))

    for x, y, z, _name in blocks:
        if min_x is None:
            min_x = max_x = x
            min_y = max_y = y
            min_z = max_z = z
        else:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)

    if min_x is None:
        raise RuntimeError("No blocks found to convert.")

    prefab_blocks = []
    for x, y, z, name in blocks:
        prefab_blocks.append(
            {
                "x": x - min_x,
                "y": y - min_y,
                "z": z - min_z,
                "name": name,
            }
        )

    prefab = {
        "version": 8,
        "blockIdVersion": 8,
        "anchorX": 0,
        "anchorY": 0,
        "anchorZ": 0,
        "blocks": prefab_blocks,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prefab, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Wrote prefab with {len(prefab_blocks)} blocks to {output_path}.")
    print(
        f"Bounds: min=({min_x},{min_y},{min_z}) max=({max_x},{max_y},{max_z})"
    )
    if unknown_counts:
        print("Unmapped legacy blocks (id:data -> count):")
        for (block_id, block_data), count in sorted(
            unknown_counts.items(), key=lambda item: -item[1]
        )[:20]:
            print(f"  {block_id}:{block_data} -> {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Minecraft world into a Hytale prefab schematic."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to Minecraft world folder (contains region/) or "
            "legacy .schematic / .schem / .litematic"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output Hytale prefab json",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional mapping JSON to override block mappings",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "legacy", "modern"),
        default="auto",
        help="Which Minecraft chunk format to use (default: auto).",
    )
    parser.add_argument(
        "--default-block",
        default=None,
        help="Override default block for unmapped entries (e.g., Empty or Air)",
    )

    args = parser.parse_args()
    mapping_path = args.mapping
    if mapping_path is None:
        default_mapping = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mappings", "default.json"
        )
        if os.path.exists(default_mapping):
            mapping_path = default_mapping

    convert_world_to_prefab(
        args.input, args.output, mapping_path, args.mode, args.default_block
    )


if __name__ == "__main__":
    main()
