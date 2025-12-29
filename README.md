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

## Collaboration Workflow

To work together effectively on this project:

1.  **GitHub Access:** Ensure your team member is added as a **Collaborator** in the GitHub repository settings.
2.  **Syncing Code:** 
    - Always `git pull` before starting work to get the latest scripts.
    - Create a new branch for significant changes: `git checkout -b feature-name`.
    - `git push` your changes and create a Pull Request on GitHub.
3.  **Syncing Data (Crucial):**
    - Since `assets/`, `frames/`, and `out_dense/` are ignored by Git, you must decide how to share these.
    - **Option A (Fresh Start):** The team member runs `fetch_tiktok_assets.py` to download everything themselves.
    - **Option B (Shared Drive):** Copy the data folders to a shared drive (OneDrive/Google Drive) so both of you have the same images/videos.
    - **Note:** If you move the project or receive a folder from someone else, run `python rebuild_manifests.py` to fix any broken file paths in the CSV manifests.
4.  **Environment:** Always use a virtual environment as described in the Setup section to avoid dependency conflicts.

