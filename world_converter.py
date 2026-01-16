#!/usr/bin/env python3
import argparse
import os

from converter import run_world_conversion


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Minecraft map into a Hytale world folder."
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
