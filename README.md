# 投影片批量修圖（網頁版）— Render 部署說明

打開網頁、拖曳多張投影片照片，系統會交給 Gemini 修圖（拉正、去雜物、去橫紋、白底純白、色彩優化），
完成後一次下載 **960×560 JPG** 的 ZIP 檔，裡面也附對照圖和處理紀錄。

```
slide_fixer_web/
├─ app.py            網頁程式（Flask）
├─ slide_core.py     修圖核心（呼叫 Gemini、轉尺寸、做對照圖）
├─ prompt.txt        修圖指令；要調整修圖方式改這個檔
├─ templates/        網頁畫面
├─ requirements.txt  套件清單
└─ render.yaml       Render 部署設定
```

---

## 一、部署到 Render

### 1. 上傳到 GitHub
把整個 `slide_fixer_web` 資料夾建成一個 GitHub repository（設成 Private 即可），然後 push 上去。

### 2. 在 Render 建立服務
**方法 A：Blueprint（推薦，設定自動帶入）**
1. Render 首頁 → **New** → **Blueprint**
2. 選剛剛的 repository，Render 會讀取 `render.yaml`
3. 畫面會要求填兩個值：
   - `GEMINI_API_KEY`：你的 Gemini API Key
   - `APP_PASSWORD`：網頁登入密碼（自己訂）
4. 按 **Apply**，等 3～5 分鐘部署完成

**方法 B：手動建立 Web Service**
- Runtime：Python 3
- Build Command：`pip install -r requirements.txt`
- Start Command：`gunicorn app:app --workers 1 --threads 8 --timeout 180`
- Environment 加入：`GEMINI_API_KEY`、`APP_PASSWORD`、`SECRET_KEY`（隨便一串長亂碼）、`PYTHON_VERSION=3.12.7`

### 3. 使用
打開 Render 給的網址（例如 `https://slide-fixer.onrender.com`），輸入密碼即可使用。

---

## 二、環境變數

| 變數 | 說明 | 預設 |
|---|---|---|
| `GEMINI_API_KEY` | **必填**，Gemini API Key | — |
| `APP_PASSWORD` | **強烈建議設定**，沒設的話任何知道網址的人都能用掉你的 API 額度 | 無 |
| `SECRET_KEY` | 登入狀態加密用，Blueprint 會自動產生 | 自動 |
| `GEMINI_IMAGE_MODEL` | 預設選取的模型 | `gemini-2.5-flash-image` |
| `GEMINI_WORKERS` | 同時呼叫 Gemini 的張數，遇到 429 限流就調成 1 | 2 |
| `MAX_FILES` | 每批最多張數 | 30 |

改完環境變數，Render 會自動重新部署。

---

## 三、Render 免費方案要注意

- **閒置 15 分鐘會休眠**，下次打開要等 30～60 秒才會醒來。
- **結果只保留 3 小時**，伺服器重啟或重新部署也會清掉，處理完記得馬上下載 ZIP。
- 工作進度存在記憶體裡，所以 Start Command 的 `--workers` 一定要是 **1**（用 `--threads` 處理多人同時使用）。
- 處理途中請不要關掉網頁。網頁每幾秒會詢問一次進度，伺服器才不會在處理途中休眠。

---

## 四、在自己電腦上測試

```
pip install -r requirements.txt
set GEMINI_API_KEY=你的KEY
set APP_PASSWORD=1234
python app.py
```
打開 http://127.0.0.1:5000

---

## 五、注意事項

1. **AI 會重畫整張圖。** 處理完後點縮圖看對照圖，核對文字與數據；小字英文標註和照片最容易被改掉。
2. **960×560 不是 16:9**。程式會請 Gemini 輸出 16:9，再縮放到 960×560，垂直方向約拉伸 3.7%，看不出來，也不會裁掉內容。
3. **圖片會上傳到 Google 處理**，公司內部技術資料請先確認資安規定。
4. Gemini 影像模型通常要先在 Google AI Studio **開啟計費**才能用 API，每張圖都會產生費用。
