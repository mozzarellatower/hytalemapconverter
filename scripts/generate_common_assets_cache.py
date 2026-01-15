#!/usr/bin/env python3
import argparse
import hashlib
import os
from pathlib import Path

def iter_files(common_dir):
    ignored = {'.DS_Store', 'Thumbs.db'}
    for path in common_dir.rglob('*'):
        if path.is_dir():
            continue
        if path.name in ignored:
            continue
        if path.name.endswith('.hash'):
            continue
        yield path

def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(
        description='Generate CommonAssetsIndex.hashes and CommonAssetsIndex.cache.'
    )
    parser.add_argument(
        '--root',
        required=True,
        help='Asset pack root containing Common/ (e.g., serverexample or extracted Assets.zip root).'
    )
    parser.add_argument(
        '--hashes',
        default=None,
        help='Output path for CommonAssetsIndex.hashes (default: <root>/CommonAssetsIndex.hashes).'
    )
    parser.add_argument(
        '--cache',
        default=None,
        help='Output path for CommonAssetsIndex.cache (default: <root>/CommonAssetsIndex.cache).'
    )

    args = parser.parse_args()
    root = Path(args.root)
    common_dir = root / 'Common'
    if not common_dir.is_dir():
        raise SystemExit(f'Missing Common directory: {common_dir}')

    hashes_path = Path(args.hashes) if args.hashes else root / 'CommonAssetsIndex.hashes'
    cache_path = Path(args.cache) if args.cache else root / 'CommonAssetsIndex.cache'

    entries = []
    for path in iter_files(common_dir):
        rel = path.relative_to(common_dir).as_posix()
        digest = sha256_file(path)
        mtime = int(path.stat().st_mtime)
        entries.append((rel, digest, mtime))

    entries.sort(key=lambda e: e[0])

    hashes_lines = ['VERSION=0']
    cache_lines = ['VERSION=1']
    for rel, digest, mtime in entries:
        hashes_lines.append(f'{digest} {rel}')
        cache_lines.append(f'{digest} {mtime} {rel}')

    hashes_path.write_text('\n'.join(hashes_lines) + '\n', encoding='utf-8')
    cache_path.write_text('\n'.join(cache_lines) + '\n', encoding='utf-8')

    print(f'Wrote {len(entries)} entries to {hashes_path}')
    print(f'Wrote {len(entries)} entries to {cache_path}')

if __name__ == '__main__':
    main()
