#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini 投影片批量修圖程式
--------------------------------
把資料夾內所有翻拍的投影片照片送到 Gemini 影像模型修復
（拉正、去雜物、去橫紋、白底純白、色彩優化），
再統一輸出成指定尺寸的 JPG（預設 960x560）。

用法：
    python gemini_slide_fixer.py -i 原圖資料夾 -o 輸出資料夾
    python gemini_slide_fixer.py -i in -o out --workers 3 --compare
    python gemini_slide_fixer.py -i in -o out --model gemini-3-pro-image-preview

API Key：設定環境變數 GEMINI_API_KEY，或用 --api-key 傳入。
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError:  # 讓 --help 在沒裝套件時也能顯示
    genai = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
RETRY_CODES = {429, 500, 502, 503, 504}
DEFAULT_MODEL = "gemini-2.5-flash-image"
MAX_UPLOAD_EDGE = 2048  # 上傳前把長邊縮到這個大小，省流量與費用

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


@dataclass
class Result:
    src: Path
    status: str          # 成功 / 略過 / 失敗
    seconds: float = 0.0
    message: str = ""
    output: str = ""


# ---------------------------------------------------------------- 影像工具
def load_image(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)  # 手機照片的旋轉資訊
    im = im.convert("RGB")
    if max(im.size) > MAX_UPLOAD_EDGE:
        im.thumbnail((MAX_UPLOAD_EDGE, MAX_UPLOAD_EDGE), Image.LANCZOS)
    return im


def to_target_size(im: Image.Image, size: tuple[int, int], mode: str) -> Image.Image:
    """把模型輸出轉成最終尺寸。
    stretch：直接縮放到目標尺寸（16:9→960x560 只會垂直拉伸約 3.7%，肉眼看不出，畫面不會被裁掉）
    crop   ：等比例縮放後裁掉多出來的邊
    pad    ：等比例縮放後補白邊
    """
    im = im.convert("RGB")
    if mode == "stretch":
        return im.resize(size, Image.LANCZOS)
    if mode == "crop":
        return ImageOps.fit(im, size, Image.LANCZOS)
    if mode == "pad":
        return ImageOps.pad(im, size, Image.LANCZOS, color=(255, 255, 255))
    raise ValueError(f"未知的 fit 模式：{mode}")


def _font(size: int):
    """回傳 (字型, 是否支援中文)。Render 等雲端主機通常沒有中文字型，就改用英文標籤。"""
    for name in ("msjh.ttc", "msjhbd.ttc", "NotoSansCJK-Regular.ttc",
                 "NotoSansCJKtc-Regular.otf", "PingFang.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        try:
            return ImageFont.truetype(name, size), True
        except OSError:
            continue
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size), False
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size), False
    except TypeError:
        return ImageFont.load_default(), False


def make_compare(before: Image.Image, after: Image.Image, size: tuple[int, int]) -> Image.Image:
    """原圖與修圖結果左右並排，方便檢查 AI 有沒有改到文字。"""
    w, h = size
    b = to_target_size(before, size, "pad")
    bar = 36
    canvas = Image.new("RGB", (w * 2 + 10, h + bar), (40, 40, 40))
    canvas.paste(b, (0, bar))
    canvas.paste(after, (w + 10, bar))
    d = ImageDraw.Draw(canvas)
    f, cjk = _font(22)
    left = "原圖 Before" if cjk else "Before"
    right = "修圖後 After（請核對文字與數據）" if cjk else "After - check text & numbers"
    d.text((12, 6), left, fill=(255, 255, 255), font=f)
    d.text((w + 22, 6), right, fill=(255, 220, 90), font=f)
    return canvas


# ---------------------------------------------------------------- Gemini 呼叫
def build_config(model: str, aspect: str, image_size: str | None):
    img_cfg = {"aspect_ratio": aspect}
    # image_size（1K/2K/4K）只有 Gemini 3 系列影像模型支援
    if image_size and model.startswith("gemini-3"):
        img_cfg["image_size"] = image_size
    return types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(**img_cfg),
    )


def call_gemini(client, model: str, prompt: str, im: Image.Image, config,
                retries: int) -> Image.Image:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(
                model=model, contents=[prompt, im], config=config)
            texts = []
            for cand in (resp.candidates or []):
                if not cand.content or not cand.content.parts:
                    if getattr(cand, "finish_reason", None):
                        texts.append(f"finish_reason={cand.finish_reason}")
                    continue
                for part in cand.content.parts:
                    if part.inline_data and part.inline_data.data:
                        return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                    if part.text:
                        texts.append(part.text.strip())
            fb = getattr(resp, "prompt_feedback", None)
            if fb and getattr(fb, "block_reason", None):
                texts.append(f"block_reason={fb.block_reason}")
            raise RuntimeError("模型沒有回傳圖片。" + (" 回覆：" + " | ".join(texts)[:300] if texts else ""))
        except genai_errors.APIError as e:
            last_err = e
            if e.code in RETRY_CODES and attempt < retries:
                wait = min(60, 2 ** attempt * 3) + random.uniform(0, 2)
                log(f"    ⚠ API {e.code}，{wait:.0f} 秒後重試（第 {attempt + 1}/{retries} 次）")
                time.sleep(wait)
                continue
            raise
        except RuntimeError as e:
            # 沒回圖片（偶發）也重試一次
            last_err = e
            if attempt < min(retries, 1):
                time.sleep(2)
                continue
            raise
    raise last_err  # pragma: no cover


# ---------------------------------------------------------------- 單張處理
def process_one(src: Path, args, client, prompt: str, config, out_dir: Path,
                cmp_dir: Path | None, idx: int, total: int) -> Result:
    out_path = out_dir / f"{src.stem}{args.suffix}.jpg"
    if out_path.exists() and not args.overwrite:
        log(f"[{idx}/{total}] 略過（已存在）：{src.name}")
        return Result(src, "略過", output=str(out_path), message="輸出檔已存在")

    t0 = time.time()
    try:
        original = load_image(src)
        log(f"[{idx}/{total}] 處理中：{src.name}")
        fixed = call_gemini(client, args.model, prompt, original, config, args.retries)
        final = to_target_size(fixed, args.size, args.fit)
        final.save(out_path, "JPEG", quality=args.quality, optimize=True, subsampling=0)
        if cmp_dir is not None:
            make_compare(original, final, args.size).save(
                cmp_dir / f"{src.stem}_對照.jpg", "JPEG", quality=88)
        dt = time.time() - t0
        log(f"[{idx}/{total}] ✔ 完成：{out_path.name}（{dt:.1f} 秒）")
        return Result(src, "成功", dt, f"模型輸出 {fixed.size[0]}x{fixed.size[1]}", str(out_path))
    except Exception as e:  # noqa: BLE001
        dt = time.time() - t0
        msg = f"{type(e).__name__}: {e}"
        log(f"[{idx}/{total}] ✘ 失敗：{src.name} → {msg[:200]}")
        return Result(src, "失敗", dt, msg)


# ---------------------------------------------------------------- 主程式
def parse_size(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().replace("×", "x").split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("尺寸格式要像 960x560")


def collect_images(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Gemini 投影片批量修圖（輸出固定尺寸 JPG）")
    ap.add_argument("-i", "--input", type=Path, default=here / "input", help="原圖資料夾（預設 ./input）")
    ap.add_argument("-o", "--output", type=Path, default=here / "output", help="輸出資料夾（預設 ./output）")
    ap.add_argument("--prompt", type=Path, default=here / "prompt.txt", help="修圖指令檔（預設 ./prompt.txt）")
    ap.add_argument("--model", default=os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_MODEL),
                    help=f"Gemini 影像模型 ID（預設 {DEFAULT_MODEL}）")
    ap.add_argument("--api-key", default=None, help="Gemini API Key（建議改用環境變數 GEMINI_API_KEY）")
    ap.add_argument("--size", type=parse_size, default=(960, 560), help="輸出尺寸，預設 960x560")
    ap.add_argument("--aspect", default="16:9", help="要求模型輸出的長寬比，預設 16:9")
    ap.add_argument("--image-size", default="2K", help="Gemini 3 系列的輸出解析度 1K/2K/4K（2.5 會忽略）")
    ap.add_argument("--fit", choices=["stretch", "crop", "pad"], default="stretch",
                    help="轉成最終尺寸的方式：stretch(預設,不裁切)/crop/pad")
    ap.add_argument("--quality", type=int, default=95, help="JPG 品質 1-100，預設 95")
    ap.add_argument("--suffix", default="_960x560", help="輸出檔名後綴，預設 _960x560")
    ap.add_argument("--workers", type=int, default=3, help="同時處理張數，預設 3（免費額度建議 1）")
    ap.add_argument("--retries", type=int, default=4, help="遇到限流/伺服器錯誤時重試次數")
    ap.add_argument("--compare", action="store_true", help="另存原圖/修圖左右對照圖，方便核對文字")
    ap.add_argument("--overwrite", action="store_true", help="輸出檔已存在時仍重新處理")
    ap.add_argument("--recursive", action="store_true", help="包含子資料夾")
    args = ap.parse_args(argv)

    if genai is None:
        print("缺少套件，請先執行：pip install -r requirements.txt")
        return 2

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("找不到 API Key。請設定環境變數 GEMINI_API_KEY，或用 --api-key 傳入。")
        return 2
    if not args.input.is_dir():
        print(f"找不到原圖資料夾：{args.input}")
        return 2
    if not args.prompt.is_file():
        print(f"找不到指令檔：{args.prompt}")
        return 2

    prompt = args.prompt.read_text(encoding="utf-8").strip()
    images = collect_images(args.input, args.recursive)
    if not images:
        print(f"資料夾內沒有圖片：{args.input}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    cmp_dir = None
    if args.compare:
        cmp_dir = args.output / "_對照圖"
        cmp_dir.mkdir(exist_ok=True)

    client = genai.Client(api_key=api_key)
    config = build_config(args.model, args.aspect, args.image_size)

    total = len(images)
    print(f"模型：{args.model}｜共 {total} 張｜輸出 {args.size[0]}x{args.size[1]} JPG｜同時 {args.workers} 張")
    print(f"輸出資料夾：{args.output.resolve()}\n")

    t0 = time.time()
    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(process_one, p, args, client, prompt, config,
                            args.output, cmp_dir, i, total)
                for i, p in enumerate(images, 1)]
        for f in as_completed(futs):
            results.append(f.result())

    results.sort(key=lambda r: str(r.src))
    log_path = args.output / "處理紀錄.csv"
    with open(log_path, "w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig 讓 Excel 正確顯示中文
        w = csv.writer(fh)
        w.writerow(["原檔", "狀態", "秒數", "輸出檔", "說明", "模型"])
        for r in results:
            w.writerow([r.src.name, r.status, f"{r.seconds:.1f}", r.output, r.message, args.model])

    ok = sum(r.status == "成功" for r in results)
    skip = sum(r.status == "略過" for r in results)
    fail = sum(r.status == "失敗" for r in results)
    print(f"\n完成：成功 {ok}、略過 {skip}、失敗 {fail}，總耗時 {time.time() - t0:.0f} 秒")
    print(f"處理紀錄：{log_path.resolve()}")
    if fail:
        print("失敗的圖片可直接再執行一次，已完成的會自動略過。")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
