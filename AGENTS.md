# Agent Notes (Codex/Claude)

This repo includes reverse-engineering output for Hytale assets plus the Minecraft -> Hytale converters.
Use this as a quick handoff for anyone continuing analysis.

## Key locations
- Decompiled Hytale server sources: `tmp_decompile/out/`
- CFR decompiler jar: `tmp_decompile/cfr.jar`
- Source jar analyzed: `serverexample/Server/HytaleServer.jar`
- Asset pack used for reference: `serverexample/Assets.zip`
- Extracted asset store paths + extensions + dependency order: `asset_store_paths.json`
- Converters: `converter.py` (world), `schematic_converter.py` (prefab), wrappers `world_converter.py`, `schematic_converter_legacy.py`, `schematic_converter_modern.py`
- Docs: `docs/USAGE.md`, `docs/summary.md`, `docs/plan.md`
- Asset cache generator: `scripts/generate_common_assets_cache.py`

## Asset system findings (summary)
- Packs are registered in `com.hypixel.hytale.server.core.asset.AssetModule`.
  - `.zip`/`.jar` packs are mounted as immutable filesystems.
  - Non-zip packs are treated immutable if `CommonAssetsIndex.hashes` exists.
- Common assets live under `Common/` and load order is:
  1) `CommonAssetsIndex.hashes` if present (hash-only load)
  2) `CommonAssetsIndex.cache` if present (timestamp-validated)
  3) Full walk of `Common/` (reads file bytes and hashes)
- Common asset hash is SHA-256 of file bytes; asset names normalize `\\` to `/`.
- Server assets are JSON (or custom extensions) loaded from `Server/<path>` for each asset store.
  - The registry block in `AssetRegistryLoader` defines each store's path, extension, and dependency order.

## Decompiled files to read first
- `tmp_decompile/out/com/hypixel/hytale/server/core/asset/AssetModule.java`
- `tmp_decompile/out/com/hypixel/hytale/server/core/asset/common/CommonAssetModule.java`
- `tmp_decompile/out/com/hypixel/hytale/server/core/asset/AssetRegistryLoader.java`
- `tmp_decompile/out/com/hypixel/hytale/server/core/asset/HytaleAssetStore.java`
- `tmp_decompile/out/com/hypixel/hytale/server/core/asset/common/CommonAsset.java`

## Converter status
- World conversion tested on `minecraftmaps/Lectus` and outputs `output/lectus_hytale`.
- Legacy prefab conversion works on Lectus; modern prefab is expected to yield zero blocks on legacy worlds.

## Tips
- If extending asset analysis, prefer CFR to decompile specific packages:
  - Example: `java -jar tmp_decompile/cfr.jar serverexample/Server/HytaleServer.jar --outputdir tmp_decompile/out --jarfilter \"com/hypixel/hytale/server/core/asset/.*|com/hypixel/hytale/assetstore/.*\"`
- When updating the asset path table, regenerate `asset_store_paths.json` from `AssetRegistryLoader.java`.
