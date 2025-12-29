# TikTok Content Analysis

This repository contains tools for analyzing TikTok content performance, brand alignment, and creator compliance.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bcali/tiktok-content-analysis.git
   cd tiktok-content-analysis
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **External Dependencies:**
   - **Tesseract OCR:** To enable overlay text detection, install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and ensure it's in your PATH, or provide the path via `--tesseract_cmd`.

## Usage

See `CMDs.txt` for example commands.

### Scoring TikTok Videos
Run the scoring script with your scraper CSV and brand configuration:
```bash
python tiktok_scoring_v3.py --posts your_data.csv --brand_json brand.json --out analysis_out
```

### Generating Executive Reports
Generate a PowerPoint executive summary:
```bash
python build_exec_report.py --scoring analysis_out/scoring_summary.csv --out Exec_Summary.pptx
```

### Dashboard
To start the update helper for the dashboard:
```bash
start_dashboard.cmd
```
(Or run `python Scripts/dashboard_update_server.py` directly)

## Project Structure
- `Scripts/`: Supplementary processing scripts.
- `assets/`: Icons, fonts, and covers.
- `brand.json`: Brand color and palette configuration.
- `out_dense/`: Default output folder for analysis results.
- `frames/`: (Local only) Extracted video frames.

## Collaboration
- **Paths:** All scripts have been updated to use relative paths. Avoid using absolute paths when contributing.
- **Data:** Large binary files (videos, frames, assets) are excluded from Git via `.gitignore`. Share these via a separate data sync if necessary.
- **Manifests:** Files like `frames_manifest.csv` might contain paths that need regeneration if the project folder is moved. Use `rebuild_manifests.py` (if available) or rerun extraction.

