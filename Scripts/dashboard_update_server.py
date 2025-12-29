#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, subprocess, shlex
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, request, redirect, make_response

app = Flask(__name__)

CONFIG = {}

def load_config(path):
    global CONFIG
    with open(path, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)

def shutil_which(cmd):
    from shutil import which
    return which(cmd)

def py_exe():
    # Prefer 'py' (Windows launcher), else 'python'
    return CONFIG.get("python") or ("py" if shutil_which("py") else "python")

def run_cmd(cmd_list, cwd=None):
    """Run a command list, capture stdout/stderr, return (rc, text)."""
    try:
        p = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True, shell=False)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode, out
    except Exception as e:
        return 1, f"[EXCEPTION] {e}"

def ensure_deps():
    # Ensure Flask (this server) and core libs for the pipeline.
    to_install = []
    for imp in ["flask","pandas","numpy","python-pptx","matplotlib"]:
        try:
            __import__(imp)
        except Exception:
            to_install.append(imp)
    if to_install:
        rc, out = run_cmd([py_exe(), "-m", "pip", "install", *to_install])
        return rc == 0, out
    return True, "All deps present."

def build_cmds():
    sdir = CONFIG["scripts_dir"]
    out_root = CONFIG["out_root"]
    scoring_csv = os.path.join(out_root, "scoring_summary.csv")

    cmds = []

    # 1) Scoring (assumes dataset scraped)
    if CONFIG.get("run_scoring", True):
        score_script = os.path.join(sdir, "tiktok_scoring_v3.py")
        cmd1 = [py_exe(), score_script,
                "--posts", CONFIG["dataset_csv"],
                "--brand_json", CONFIG["brand_json"],
                "--videos", CONFIG.get("videos_dir",""),
                "--out", out_root]
        if CONFIG.get("frames_manifest"):
            cmd1 += ["--frames-manifest", CONFIG["frames_manifest"]]
        cmds.append(("Scoring", cmd1))

    # 2) PPT (optional)
    if CONFIG.get("run_ppt", True):
        ppt_script = os.path.join(sdir, "build_exec_report_v2.py")
        out_ppt = os.path.join(out_root, "reports", "exec_summary", "Exec_Summary.pptx")
        os.makedirs(os.path.dirname(out_ppt), exist_ok=True)
        cmd2 = [py_exe(), ppt_script, "--scoring", scoring_csv]
        if CONFIG.get("covers_manifest"):
            cmd2 += ["--covers", CONFIG["covers_manifest"]]
        cmd2 += ["--out", out_ppt]
        cmds.append(("PowerPoint", cmd2))

    # 3) HTML dashboard
    dash_script = os.path.join(sdir, "build_dashboard_html_v6.py")  # Updated to v6
    out_dir = CONFIG["dashboard_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cmd3 = [py_exe(), dash_script,
            "--scoring", scoring_csv,
            "--brand_json", CONFIG["brand_json"]]
    if CONFIG.get("covers_manifest"):
        cmd3 += ["--covers", CONFIG["covers_manifest"]]
    if CONFIG.get("covers_root"):
        cmd3 += ["--covers_root", CONFIG["covers_root"]]
    if CONFIG.get("frames_manifest"):
        cmd3 += ["--frames_manifest", CONFIG["frames_manifest"]]
    if CONFIG.get("frames_root"):
        cmd3 += ["--frames_root", CONFIG["frames_root"]]
    if CONFIG.get("logo"):
        cmd3 += ["--logo", CONFIG["logo"]]
    if CONFIG.get("fonts_dir"):
        cmd3 += ["--fonts_dir", CONFIG["fonts_dir"]]
    cmd3 += ["--out_dir", out_dir]
    cmds.append(("Dashboard", cmd3))

    return cmds

@app.after_request
def add_cors(resp):
    # allow the file:// or other origins to POST to /update if needed
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

@app.route("/")
def home():
    # Always show the control page first
    return redirect("/control")

@app.route("/report/")
@app.route("/report/<path:path>")
def report(path="index.html"):
    base = CONFIG["dashboard_dir"]
    if not os.path.isdir(base):
        return "Dashboard not generated yet. Use /control to run Update.", 404
    return send_from_directory(base, path)

@app.route("/control")
def control():
    # Minimal UI to run Update and see logs
    return """
<!doctype html><html><head><meta charset="utf-8"><title>TikTok Dashboard Control</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;padding:20px;max-width:900px;margin:auto;background:#0b1220;color:#e8eef9}
h1,h2{margin:0 0 12px 0}
.card{background:#10192e;padding:16px;border-radius:10px;margin:12px 0;border:1px solid #1e2a47}
.btn{font-size:16px;padding:10px 18px;border-radius:10px;border:0;background:#2dd4bf;color:#083344;cursor:pointer}
.btn:disabled{opacity:.6;cursor:not-allowed}
pre{white-space:pre-wrap;background:#0a0f1f;color:#dbe7ff;padding:12px;border-radius:8px;min-height:80px}
a{color:#8fb9ff}
</style>
</head><body>
<h1>TikTok Dashboard Control</h1>
<div class="card">
  <p>Click <b>Update</b> to (re)build scoring, PowerPoint, and the dashboard.</p>
  <button class="btn" id="btnUpdate">Update</button>
  <a href="/report/" id="lnkReport" style="margin-left:12px">Open Dashboard</a>
</div>
<div class="card">
  <h2>Log</h2>
  <pre id="log">Idle.</pre>
</div>
<script>
const btn=document.getElementById('btnUpdate');
const log=document.getElementById('log');
btn.addEventListener('click', async ()=>{
  btn.disabled=true; btn.textContent='Updating...'; log.textContent='Running update...';
  try{
    const r=await fetch('/update',{method:'POST'});
    if(!r.ok) throw new Error('HTTP '+r.status);
    const j=await r.json();
    log.textContent=j.log||'';
    btn.textContent=j.ok?'Done':'Update failed';
  }catch(e){
    log.textContent='Error: '+e.toString();
    btn.textContent='Update';
  }finally{
    btn.disabled=false;
  }
});
</script>
</body></html>
"""

@app.route("/update", methods=["POST","OPTIONS"])
def update():
    if request.method == "OPTIONS":
        return make_response("", 204)

    ok, dep_log = ensure_deps()
    logs = [f"== Ensuring dependencies ==\n{dep_log}\n"]

    if not ok:
        return jsonify({"ok": False, "log": "\n".join(logs)})

    cmds = build_cmds()
    for title, cmd in cmds:
        logs.append(f"\n== {title} ==\n$ " + " ".join(shlex.quote(c) for c in cmd))
        rc, out = run_cmd(cmd, cwd=os.path.dirname(cmd[1]))
        logs.append(out.strip())
        if rc != 0:
            logs.append(f"[{title}] FAILED (rc={rc})")
            return jsonify({"ok": False, "log": "\n".join(logs)})
        else:
            logs.append(f"[{title}] OK")

    logs.append("\nAll done at " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    return jsonify({"ok": True, "log": "\n".join(logs)})

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: dashboard_update_server.py <path_to_update_config.json>")
        sys.exit(2)
    load_config(sys.argv[1])
    app.run(host="127.0.0.1", port=8777, debug=False)
