# Bed Wars (Minecraft) -> Hytale Converter Plan

## Goals
- Build a Python converter that ingests a Minecraft Bed Wars map (Anvil/level.dat + region files) and outputs a Hytale-compatible world/assets package.
- Use `minecraftmaps/Lectus` as the reference input and validate results against the `serverexample` assets/server.
- Produce a repeatable CLI workflow for future Bed Wars maps.

## Assumptions (to validate early)
- Hytale server world format can be derived from `serverexample` (likely a world/universe directory structure + metadata + assets pack).
- Bed Wars maps are static builds; no complex redstone or entity logic is required beyond basic block/state conversion.
- We can ignore dimensions beyond the overworld unless required by the map.

## Phase 1: Discovery and Format Recon
1. **Inventory Minecraft input**
   - Read `minecraftmaps/Lectus/level.dat` for world metadata (spawn, dimension info, version).
   - Enumerate `minecraftmaps/Lectus/region/*.mca` to estimate chunk bounds and size.
   - Identify any Bed Wars-specific markers (e.g., player spawns, shop locations, resource generators) via block types or NBT data if present.

2. **Inventory Hytale output format**
   - Inspect `serverexample/universe` and `serverexample/Assets.zip` to identify:
     - World format (files, metadata, chunk storage, compression).
     - Expected asset references (blocks, materials, models, textures).
     - Any sample world that can be treated as a template.
   - Determine which files must be generated vs. copied from template.

3. **Decide conversion targets**
   - Minimum viable output: terrain blocks + static structures.
   - Optional: entities, chests/loot, spawn points, teams/bed positions.
   - Define “done” criteria for the example map (loads into serverexample without crash + visually correct terrain).

## Phase 2: Mapping Design
1. **Block/state mapping table**
   - Build a mapping table from Minecraft block IDs/states to Hytale block IDs or materials.
   - Include defaults/fallbacks (unknown block -> nearest material or placeholder).

2. **Coordinate and chunk conversion**
   - Map Minecraft chunk coordinates to Hytale chunk coordinates (verify axis orientation and height ranges).
   - Handle vertical bounds (Minecraft 0–255 vs. Hytale’s range if different).

3. **Bed Wars semantics (if available)**
   - If data exists, record spawn points, bed positions, and resource generators.
   - Define how to encode these into Hytale (markers, prefabs, or simple notes in metadata).

## Phase 3: Implementation
1. **Parser for Minecraft Anvil format**
   - Read `level.dat` (NBT) for metadata.
   - Parse `.mca` files to extract chunk sections and block palettes.
   - Normalize blocks to a simple internal structure: `[(x, y, z) -> block_id/state]` with chunk bounds.

2. **Hytale writer**
   - Use `serverexample` as a template; copy required scaffolding.
   - Generate Hytale chunk data files/regions from internal representation.
   - Write or update world metadata (seed, spawn, bounds) if required.

3. **CLI and config**
   - `converter.py` with args: input path, output path, mapping file, dimension selection.
   - Optional JSON/YAML config for mappings and overrides.

## Phase 4: Validation
1. **Dry-run checks**
   - Validate block counts, chunk bounds, missing mappings.
   - Emit a report of unknown blocks and conversion coverage.

2. **Server load test**
   - Drop converted world into `serverexample` and boot server.
   - Verify map loads without errors and the main structures are in the correct location.

3. **Documentation**
   - Update README or create `USAGE.md` with steps, example commands, limitations.

## Deliverables
- `converter.py` (or a small Python package) with CLI.
- `mappings/` with a default Minecraft->Hytale map.
- `plan.md` (this file) and a short usage doc.

## Risks / Unknowns
- Hytale file format may not be obvious from `serverexample`; might require reverse-engineering or a template-based approach.
- Some Minecraft blocks may not have direct Hytale equivalents.
- Bed Wars metadata (spawns/shops) might not be present in world data and may need manual tagging.

## Next Actions (immediate)
- Inspect `serverexample` world files and assets for format details.
- Parse the Lectus map to list all block IDs/states used.
- Draft the initial block mapping table.
