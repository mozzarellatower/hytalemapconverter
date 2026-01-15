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
  --template serverexample/universe/worlds/default
```

### Mapping
The converter loads `mappings/default.json` automatically when present. To override mappings:

```bash
python3 world_converter.py \
  --input minecraftmaps/Lectus \
  --output output/lectus_hytale \
  --template serverexample/universe/worlds/default \
  --mapping mappings/default.json
```

### Notes
- This handles legacy Minecraft chunks (1.8-era `Blocks`/`Data`/`Add`).
- Entities, tile entities, and Bed Wars gameplay metadata are not converted yet.

## Minecraft -> Hytale Prefab Schematic
Use `schematic_converter.py` to export a Hytale prefab schematic JSON from a Minecraft world folder:

```bash
python3 schematic_converter.py \
  --input minecraftmaps/Lectus \
  --output output/lectus.prefab.json

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
```
