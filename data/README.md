# Data

CS2 demo files (`.dem`) are **not tracked in this repository** because:
- Each file is 100–200 MB
- They are copyrighted recordings owned by Valve / the tournament organisers

## How to Get Demo Files

Professional match demos are publicly downloadable from [HLTV.org](https://www.hltv.org):

1. Go to any match page on HLTV (e.g., a Major match)
2. Click the **"Demo"** tab near the top of the match page
3. Download the `.dem` file for the map you want
4. Place it inside your configured `raw_dems/` folder on Google Drive

## Folder Setup

After downloading, your Drive should look like:
```
MyDrive/
└── cs2-btp/
    ├── raw_dems/
    │   ├── vitality_vs_spirit_dust2.dem
    │   ├── natus_vincere_vs_g2_mirage.dem
    │   └── ...
    └── processed/          ← created automatically by the pipeline
```

## Recommended Sources

| Source | URL | Notes |
|--------|-----|-------|
| HLTV.org | https://www.hltv.org | Best source for professional demos |
| CS2 in-game theatre | `cs2` → Watch → Downloads | Your own matchmaking replays |

## Notes on Demo Quality

- Use **64-tick** competitive demos for consistent tick rates
- Avoid very old demos (pre-2024) — the `demoparser2` event schema may differ
- The pipeline handles corrupted or incomplete demos gracefully via `try/except` wrappers