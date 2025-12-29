# TikTok Content Analysis

This repository contains tools for analyzing TikTok content performance, brand alignment, and creator compliance.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bcali/tiktok-content-analysis.git
   cd tiktok-content-analysis
   ```

2. **Install dependencies:**
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
python Scripts/tiktok_scoring_v3.py --posts Data/your_data.csv --brand_json brand.json --out out_dense
```

### Generating Executive Reports
Generate a PowerPoint executive summary:
```bash
python Scripts/build_exec_report_v2.py --scoring out_dense/scoring_summary.csv --out Exec_Summary.pptx
```

### Dashboard
To start the update helper for the dashboard:
```bash
start_dashboard.cmd
```

## Project Structure
- `Scripts/`: Main analysis and processing scripts.
- `Data/`: Source datasets and CSVs from scrapers.
- `Docs/`: Project documentation, PDFs, and report summaries.
- `assets/`: Icons, fonts, and covers (ignored by Git).
- `out_dense/`: Output folder for analysis results (ignored by Git).
- `frames/`: Extracted video frames (ignored by Git).
- `brand.json`: Brand color and palette configuration.

## Collaboration Workflow

To work together effectively on this project:

1.  **GitHub Access:** Ensure your team member is added as a **Collaborator** in the GitHub repository settings.
2.  **Syncing Code:** 
    - Always `git pull` before starting work to get the latest scripts.
    - Create a new branch for significant changes: `git checkout -b feature-name`.
    - `git push` your changes and create a Pull Request on GitHub.
3.  **Syncing Data (Crucial):**
    - Large binary files (videos, frames) are **NOT** in Git.
    - **Option A:** The team member runs `Scripts/fetch_tiktok_assets.py` to download everything themselves.
    - **Option B:** Copy the `assets/`, `frames/`, and `out_dense/` folders via a shared drive.
    - **Path Fixer:** If you move the project or receive data from someone else, run `python Scripts/rebuild_manifests.py` to fix any broken file paths in the CSV manifests.
4.  **Organized Paths:** All scripts use relative paths. Avoid using absolute paths when contributing.
