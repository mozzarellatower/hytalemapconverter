#!/usr/bin/env python3
import argparse
import os

from converter import convert_world


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Minecraft map into a Hytale world folder."
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
        help=(
            "Path to template Hytale world folder "
            "(e.g., serverexample/universe/worlds/default)"
        ),
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
