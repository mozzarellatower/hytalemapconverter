#!/usr/bin/env python3
import argparse
import os

from schematic_converter import convert_world_to_prefab


def main():
    parser = argparse.ArgumentParser(
        description="Convert a legacy Minecraft world into a Hytale prefab schematic."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Minecraft world folder (contains region/)",
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

    args = parser.parse_args()
    mapping_path = args.mapping
    if mapping_path is None:
        default_mapping = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mappings", "default.json"
        )
        if os.path.exists(default_mapping):
            mapping_path = default_mapping

    convert_world_to_prefab(args.input, args.output, mapping_path, mode="legacy")


if __name__ == "__main__":
    main()
