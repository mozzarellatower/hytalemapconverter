# MapConverter (Minecraft -> Hytale)

Tools for converting Minecraft maps into Hytale worlds or Hytale prefab schematics.

## What's Included
- World converter: Minecraft world -> Hytale world folder
- Prefab converter: Minecraft world -> Hytale prefab JSON
- Mapping file: `mappings/default.json` for block ID/state mapping
- Asset reverse-engineering notes and extracted store paths

## Requirements
- Python 3.9+
- Python modules: `nbtlib`, `pymongo`, `zstandard`

Install dependencies:
```bash
python3 -m pip install --break-system-packages nbtlib pymongo zstandard
```

## Usage
World conversion (drag-in Hytale world folder):
```bash
python3 world_converter.py \
  --input minecraftmaps/YourMap \
  --output output/yourmap_hytale \
  --template save/universe/worlds/default
```

Optional: create or reuse a small template cache (repo includes `template_cache.json`). If you omit both `--template` and `--template-cache`, it will default to the bundled cache:
```bash
python3 world_converter.py \
  --input minecraftmaps/YourMap \
  --output output/yourmap_hytale \
  --template save/universe/worlds/default \
  --template-cache template_cache.json
```

Example layout:
```
project-root/
  minecraftmaps/
    YourMap/
      region/
  output/
    yourmap_hytale/
  save/
    universe/
      worlds/
        default/    # template world (from a Hytale server save)
```

Prefab conversion (schematic JSON, world folder, `.schematic`, `.schem`, or `.litematic` file):
```bash
python3 schematic_converter.py \
  --input minecraftmaps/YourMap \
  --output output/yourmap.prefab.json
```

```bash
python3 schematic_converter.py \
  --input minecraftmaps/YourMap.schematic \
  --output output/yourmap.prefab.json
```

```bash
python3 schematic_converter.py \
  --input minecraftmaps/YourMap.schem \
  --output output/yourmap.prefab.json
```

```bash
python3 schematic_converter.py \
  --input minecraftmaps/YourMap.litematic \
  --output output/yourmap.prefab.json
```

Legacy/modern specific prefab conversion:
```bash
python3 schematic_converter_legacy.py \
  --input minecraftmaps/YourMap \
  --output output/yourmap_legacy.prefab.json
```

```bash
python3 schematic_converter_modern.py \
  --input minecraftmaps/YourMap \
  --output output/yourmap_modern.prefab.json
```

## Mapping
The converters load `mappings/default.json` automatically. To override:
```bash
python3 world_converter.py \
  --input minecraftmaps/YourMap \
  --output output/yourmap_hytale \
  --template save/universe/worlds/default \
  --mapping mappings/default.json
```

To avoid defaulting unknown blocks to stone, pass:
```bash
python3 world_converter.py \
  --input minecraftmaps/YourMap \
  --output output/yourmap_hytale \
  --default-block Empty
```

## Notes
- Legacy (1.8-era) chunks and modern Anvil palette chunks are supported.
- Entities and tile entities are not converted yet.

## Docs
- `docs/USAGE.md` for more details
- `docs/summary.md` for a project summary
- `docs/plan.md` for the original conversion plan
