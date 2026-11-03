# -*- coding: utf-8 -*-
"""
Gemini 投影片批量修圖 — 網頁版（Render 部署用）

環境變數：
  GEMINI_API_KEY      必填，Gemini API Key
  APP_PASSWORD        建議設定，進入網頁要輸入的密碼（避免別人用掉你的 API 額度）
  SECRET_KEY          Flask session 加密用（render.yaml 會自動產生）
  GEMINI_IMAGE_MODEL  預設模型，預設 gemini-2.5-flash-image
  GEMINI_WORKERS      同時呼叫 Gemini 的張數，預設 2
  MAX_FILES           每批最多張數，預設 30
"""
from __future__ import annotations

import csv
import io
import os
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_file, session, url_for)
from werkzeug.utils import secure_filename

import slide_core as core

BASE = Path(__file__).resolve().parent
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/tmp/slide_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
DEFAULT_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", core.DEFAULT_MODEL)
WORKERS = int(os.environ.get("GEMINI_WORKERS", "2"))
MAX_FILES = int(os.environ.get("MAX_FILES", "30"))
JOB_TTL = 3 * 3600  # 工作檔保留 3 小時後自動刪除
OUT_SIZE = (960, 560)

MODELS = {
    "gemini-2.5-flash-image": "Gemini 2.5 Flash Image（快、便宜）",
    "gemini-3.1-flash-image-preview": "Gemini 3.1 Flash Image（較新）",
    "gemini-3-pro-image-preview": "Gemini 3 Pro Image（品質最好、較貴）",
}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 單次上傳上限 200MB

_pool = ThreadPoolExecutor(max_workers=max(1, WORKERS))
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("伺服器尚未設定 GEMINI_API_KEY")
        _client = core.genai.Client(api_key=key)
    return _client


def read_prompt() -> str:
    return (BASE / "prompt.txt").read_text(encoding="utf-8").strip()


# ------------------------------------------------------------ 登入保護
def logged_in() -> bool:
    return not APP_PASSWORD or session.get("ok") is True


@app.before_request
def guard():
    if request.endpoint in ("login", "static", "health"):
        return None
    if not logged_in():
        if request.path.startswith("/api/"):
            return jsonify(error="請先登入"), 401
        return redirect(url_for("login"))
    return None


@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return redirect(url_for("index"))
    err = ""
    if request.method == "POST":
        if secrets.compare_digest(request.form.get("password", ""), APP_PASSWORD):
            session["ok"] = True
            session.permanent = True
            return redirect(url_for("index"))
        err = "密碼錯誤"
    return render_template("login.html", error=err)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/healthz")
def health():
    return "ok"


# ------------------------------------------------------------ 頁面
@app.route("/")
def index():
    return render_template("index.html", models=MODELS, default_model=DEFAULT_MODEL,
                           max_files=MAX_FILES, has_password=bool(APP_PASSWORD),
                           key_ready=bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")))


# ------------------------------------------------------------ 批次工作
def _process(job_id: str, idx: int):
    job = _jobs[job_id]
    item = job["items"][idx]
    jdir = JOBS_DIR / job_id
    src = jdir / "in" / item["stored"]
    out_name = f"{Path(item['name']).stem}_960x560.jpg"
    item["status"] = "處理中"
    t0 = time.time()
    try:
        original = core.load_image(src)
        config = core.build_config(job["model"], "16:9", "2K")
        fixed = core.call_gemini(get_client(), job["model"], job["prompt"], original, config, retries=4)
        final = core.to_target_size(fixed, OUT_SIZE, "stretch")
        final.save(jdir / "out" / out_name, "JPEG", quality=95, optimize=True, subsampling=0)
        if job["compare"]:
            core.make_compare(original, final, OUT_SIZE).save(
                jdir / "out" / "_對照圖" / f"{Path(item['name']).stem}_對照.jpg", "JPEG", quality=88)
        item.update(status="成功", output=out_name, message="")
    except Exception as e:  # noqa: BLE001
        item.update(status="失敗", message=f"{type(e).__name__}: {e}"[:300])
    finally:
        item["seconds"] = round(time.time() - t0, 1)
        with _lock:
            job["done"] += 1
            if job["done"] == len(job["items"]):
                job["finished_at"] = time.time()
                _write_log(job_id)


def _write_log(job_id: str):
    job = _jobs[job_id]
    path = JOBS_DIR / job_id / "out" / "處理紀錄.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["原檔", "狀態", "秒數", "輸出檔", "說明", "模型"])
        for it in job["items"]:
            w.writerow([it["name"], it["status"], it.get("seconds", ""), it.get("output", ""),
                        it.get("message", ""), job["model"]])


def _unique_name(name: str, used: set[str]) -> str:
    stem, ext = Path(name).stem, Path(name).suffix
    cand, n = name, 2
    while cand.lower() in used:
        cand = f"{stem}({n}){ext}"
        n += 1
    used.add(cand.lower())
    return cand


@app.post("/api/jobs")
def create_job():
    _cleanup()
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify(error="沒有收到圖片"), 400
    if len(files) > MAX_FILES:
        return jsonify(error=f"一次最多 {MAX_FILES} 張"), 400
    model = request.form.get("model") or DEFAULT_MODEL
    if model not in MODELS:
        return jsonify(error="不支援的模型"), 400
    try:
        get_client()
    except RuntimeError as e:
        return jsonify(error=str(e)), 500

    job_id = uuid.uuid4().hex[:12]
    jdir = JOBS_DIR / job_id
    (jdir / "in").mkdir(parents=True)
    (jdir / "out" / "_對照圖").mkdir(parents=True)

    items, used = [], set()
    for i, f in enumerate(files):
        orig = Path(f.filename).name  # 保留中文檔名給輸出用
        ext = Path(orig).suffix.lower()
        if ext not in core.IMAGE_EXTS:
            continue
        display = _unique_name(orig, used)
        stored = f"{i:03d}{ext}"  # 磁碟上用安全檔名
        f.save(jdir / "in" / secure_filename(stored))
        items.append({"name": display, "stored": stored, "status": "排隊中"})
    if not items:
        shutil.rmtree(jdir, ignore_errors=True)
        return jsonify(error="沒有可處理的圖片格式（支援 jpg/png/webp/bmp/tif）"), 400

    _jobs[job_id] = {"items": items, "done": 0, "model": model,
                     "compare": request.form.get("compare", "1") == "1",
                     "prompt": read_prompt(), "created": time.time(), "finished_at": None}
    for i in range(len(items)):
        _pool.submit(_process, job_id, i)
    return jsonify(job_id=job_id, total=len(items))


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = _jobs.get(job_id) or abort(404)
    return jsonify(
        total=len(job["items"]), done=job["done"], finished=job["finished_at"] is not None,
        model=job["model"], compare=job["compare"],
        items=[{k: it.get(k) for k in ("name", "status", "output", "message", "seconds")}
               for it in job["items"]])


@app.get("/api/jobs/<job_id>/file/<path:name>")
def job_file(job_id, name):
    if job_id not in _jobs:
        abort(404)
    base = (JOBS_DIR / job_id / "out").resolve()
    p = (base / name).resolve()
    if base not in p.parents or not p.is_file():
        abort(404)
    return send_file(p, mimetype="image/jpeg", as_attachment=request.args.get("dl") == "1",
                     download_name=p.name)


@app.get("/api/jobs/<job_id>/zip")
def job_zip(job_id):
    if job_id not in _jobs:
        abort(404)
    out = JOBS_DIR / job_id / "out"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"修圖結果_{time.strftime('%Y%m%d_%H%M')}.zip")


def _cleanup():
    now = time.time()
    for jid, job in list(_jobs.items()):
        if job["finished_at"] and now - job["finished_at"] > JOB_TTL:
            shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
            _jobs.pop(jid, None)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="上傳檔案太大（單次上限 200MB）"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
