# Map Converter Usage

## Requirements
- Python 3
- Modules: `nbtlib`, `pymongo`, `zstandard`

## Minecraft -> Hytale World
Use `world_converter.py` (or `converter.py`) to turn a Minecraft world folder into a Hytale world folder using a template server world:

```bash
python3 world_converter.py \
  --input minecraftmaps/Lectus \
  --output output/lectus_hytale \
  --template save/universe/worlds/default
```

### Template cache
You can generate a small template cache JSON once, then reuse it without a full template.
This repo now includes a ready-made cache at `template_cache.json`.

```bash
python3 world_converter.py \
  --input minecraftmaps/Lectus \
  --output output/lectus_hytale \
  --template save/universe/worlds/default \
  --template-cache template_cache.json
```

After the cache exists, you can omit `--template` (or omit both and let it default to `template_cache.json`):

```bash
python3 world_converter.py \
  --input minecraftmaps/Lectus \
  --output output/lectus_hytale \
  --template-cache template_cache.json
```

If you omit `--template`, config/resources are not copied.

### Large worlds, caching, and resume
If the world is larger than 100MB, the converter will prompt to run in chunked or parallel mode and build a cache under `worldcache/`.
You can resume an interrupted run from the same folder:

```bash
python3 world_converter.py --continue
```

Prompts can be skipped with `--ignoreprompt`. To force a mode, use `--mode chunked` or `--mode parallel` and optionally `--workers 8`.

### Mapping
The converter loads `mappings/default.json` automatically when present. To override mappings:

```bash
python3 world_converter.py \
  --input minecraftmaps/Lectus \
  --output output/lectus_hytale \
  --template save/universe/worlds/default \
  --mapping mappings/default.json
```

To avoid defaulting unknown blocks to stone, use `--default-block Empty` (or `Air`):

```bash
python3 world_converter.py \
  --input minecraftmaps/Lectus \
  --output output/lectus_hytale \
  --default-block Empty
```

### Notes
- This handles legacy Minecraft chunks (1.8-era `Blocks`/`Data`/`Add`).
- Entities, tile entities, and Bed Wars gameplay metadata are not converted yet.

## Minecraft -> Hytale Prefab Schematic
Use `schematic_converter.py` to export a Hytale prefab schematic JSON from a Minecraft world folder, legacy `.schematic`, `.schem`, or `.litematic` file:

```bash
python3 schematic_converter.py \
  --input minecraftmaps/Lectus \
  --output output/lectus.prefab.json
```

```bash
python3 schematic_converter.py \
  --input minecraftmaps/Lectus.schematic \
  --output output/lectus.prefab.json
```

```bash
python3 schematic_converter.py \
  --input minecraftmaps/Lectus.schem \
  --output output/lectus.prefab.json
```

```bash
python3 schematic_converter.py \
  --input minecraftmaps/Lectus.litematic \
  --output output/lectus.prefab.json
```

### Legacy vs Modern Formats
If you want separate scripts per Minecraft format, use:

```bash
python3 schematic_converter_legacy.py \
  --input minecraftmaps/Lectus \
  --output output/lectus_legacy.prefab.json
```

```bash
python3 schematic_converter_modern.py \
  --input minecraftmaps/Lectus \
  --output output/lectus_modern.prefab.json
```

You can also set `--mode legacy` or `--mode modern` with `schematic_converter.py`.
