import os
import time
import threading
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, Response, send_from_directory

import psutil
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")  # gerar PNG sem display
import matplotlib.pyplot as plt

APP_DIR = os.environ.get("APP_DIR", "/app")
DB_PATH = os.environ.get("DB_PATH", "/app/data/metrics.db")
INTERVAL_SEC = float(os.environ.get("INTERVAL_SEC", "5"))
WINDOW_MINUTES = float(os.environ.get("WINDOW_MINUTES", "60"))
DEMO_SEED = int(os.environ.get("DEMO_SEED_DATA", "1"))  # 1 = semear dados demo se vazio
REPORT_DIR = os.environ.get("REPORT_DIR", "/app/report")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

app = Flask(__name__)

# ------------------------- DB / COLETA -------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics(
            ts TEXT PRIMARY KEY,
            cpu_percent REAL,
            mem_percent REAL,
            load_1m REAL,
            load_5m REAL,
            load_15m REAL
        );
    """)
    con.commit()
    con.close()

def insert_metric(row):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO metrics(ts, cpu_percent, mem_percent, load_1m, load_5m, load_15m)
        VALUES (?, ?, ?, ?, ?, ?)
    """, row)
    con.commit()
    con.close()

def sample_forever():
    while True:
        ts = datetime.utcnow().isoformat()
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        try:
            l1, l5, l15 = os.getloadavg()
        except (AttributeError, OSError):
            l1 = l5 = l15 = 0.0
        insert_metric((ts, cpu, mem, l1, l5, l15))
        time.sleep(INTERVAL_SEC)

def load_window_df(minutes: float):
    since = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM metrics WHERE ts >= ? ORDER BY ts ASC;",
        con, params=(since,)
    )
    con.close()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def seed_demo_if_needed():
    """Preenche o banco com ~30min de dados sintéticos caso esteja vazio (para demo imediata)."""
    if not DEMO_SEED:
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM metrics")
    n = cur.fetchone()[0]
    con.close()
    if n >= 60:  # já tem dados
        return
    now = datetime.utcnow()
    for i in range(180):  # 180 amostras de 10s ~ 30 min
        ts = (now - timedelta(seconds=10*(180-i))).isoformat()
        cpu = max(0, min(100, 30 + 10*np.sin(i/10) + np.random.randn()*2))
        mem = max(0, min(100, 45 + 5*np.cos(i/15) + np.random.randn()*1.5))
        l1 = max(0, 0.2 + 0.05*np.sin(i/20))
        l5 = max(0, 0.15 + 0.03*np.cos(i/22))
        l15 = max(0, 0.1 + 0.02*np.sin(i/25))
        insert_metric((ts, float(cpu), float(mem), float(l1), float(l5), float(l15)))

# ------------------------- IA -------------------------
def make_forecast(df: pd.DataFrame, lags: int = 5):
    for k in range(1, lags + 1):
        df[f"cpu_lag_{k}"] = df["cpu_percent"].shift(k)
    df = df.dropna().reset_index(drop=True)
    if len(df) < (lags + 10):
        return None
    X = df[[f"cpu_lag_{k}" for k in range(1, lags + 1)]]
    y = df["cpu_percent"]
    split = int(len(df)*0.8)
    if split == 0 or split >= len(df):
        return None
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if len(X_test) == 0:
        return None
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test.values)))
    ss_tot = float(np.sum((y_test.values - np.mean(y_test.values))**2))
    ss_res = float(np.sum((y_test.values - pred)**2))
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
    return {
        "df": df, "split": split,
        "y_test": y_test.values, "pred": pred,
        "mae": mae, "r2": r2
    }

def make_anomaly(df: pd.DataFrame, contamination: float = 0.03):
    if len(df) < 30:
        return None
    feats = df[["cpu_percent", "mem_percent", "load_1m", "load_5m", "load_15m"]].values
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    labels = iso.fit_predict(feats)
    out = df.copy()
    out["anomaly"] = (labels == -1).astype(int)
    return {"df": out, "rate": float(out["anomaly"].mean())}

def make_clusters(df: pd.DataFrame, k: int = 3):
    if len(df) < k*10:
        return None
    feats = df[["cpu_percent", "mem_percent", "load_1m", "load_5m", "load_15m"]].values
    Z = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Z)
    out = df.copy()
    out["cluster"] = labels
    sizes = out["cluster"].value_counts().sort_index().to_dict()
    return {"df": out, "sizes": sizes, "k": k}

# ------------------------- GRÁFICOS / RELATÓRIO -------------------------
def save_plot(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def generate_charts():
    df = load_window_df(WINDOW_MINUTES if WINDOW_MINUTES > 1 else 60)

    # 1) CPU/Mem
    fig = plt.figure()
    if not df.empty:
        plt.plot(df["ts"], df["cpu_percent"], label="CPU %")
        plt.plot(df["ts"], df["mem_percent"], label="Mem %")
        plt.xticks(rotation=30, ha="right")
        plt.legend(); plt.title("CPU e Memória (%)"); plt.tight_layout()
    save_plot(fig, os.path.join(REPORT_DIR, "cpu_mem.png"))

    # 2) Loads
    fig = plt.figure()
    if not df.empty:
        plt.plot(df["ts"], df["load_1m"], label="load_1m")
        plt.plot(df["ts"], df["load_5m"], label="load_5m")
        plt.plot(df["ts"], df["load_15m"], label="load_15m")
        plt.xticks(rotation=30, ha="right")
        plt.legend(); plt.title("Cargas (load average)"); plt.tight_layout()
    save_plot(fig, os.path.join(REPORT_DIR, "loads.png"))

    # 3) Forecast
    fig = plt.figure()
    fc = None
    if not df.empty:
        fc = make_forecast(df.copy(), lags=5)
    if fc:
        y_test, pred = fc["y_test"], fc["pred"]
        plt.plot(range(len(y_test)), y_test, label="Real (test)")
        plt.plot(range(len(pred)), pred, label="Previsto")
        plt.title(f"Forecast CPU — MAE={fc['mae']:.2f} | R2={fc['r2']:.2f}")
        plt.legend(); plt.tight_layout()
    save_plot(fig, os.path.join(REPORT_DIR, "forecast.png"))

    # 4) Anomalias
    fig = plt.figure()
    an = None
    if not df.empty:
        an = make_anomaly(df.copy(), contamination=0.03)
    if an:
        dfa = an["df"]
        plt.plot(dfa["ts"], dfa["cpu_percent"], label="CPU %")
        anomalies = dfa[dfa["anomaly"] == 1]
        if not anomalies.empty:
            plt.scatter(anomalies["ts"], anomalies["cpu_percent"], s=30, marker="x", label="Anomalia")
        plt.xticks(rotation=30, ha="right")
        plt.title(f"Detecção de Anomalias (taxa={an['rate']:.2%})")
        plt.legend(); plt.tight_layout()
    save_plot(fig, os.path.join(REPORT_DIR, "anomaly.png"))

    # 5) Clusters
    fig = plt.figure()
    cl = None
    if not df.empty:
        cl = make_clusters(df.copy(), k=3)
    if cl:
        dff = cl["df"]
        plt.scatter(dff["cpu_percent"], dff["mem_percent"], c=dff["cluster"])
        plt.xlabel("CPU %"); plt.ylabel("Mem %")
        plt.title(f"Clusters (k={cl['k']}) — tamanhos: {cl['sizes']}")
        plt.tight_layout()
    save_plot(fig, os.path.join(REPORT_DIR, "clusters.png"))

    # --------- HTML com caminhos corretos /report/img/... ---------
    html_path = os.path.join(REPORT_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório — IA Collector Cloud</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 980px; margin: auto; line-height: 1.42; }}
  h1,h2 {{ margin-top: 24px; }}
  img {{ max-width: 100%; border: 1px solid #ddd; padding: 4px; background:#fff; }}
  code {{ background:#f5f5f5; padding:2px 4px; border-radius:3px; }}
  .meta {{ color:#444 }}
  hr {{ margin: 24px 0; }}
</style>
</head><body>

<h1>Relatório — IA Collector Cloud</h1>
<p class="meta"><b>Intervalo:</b> {INTERVAL_SEC:.1f}s &nbsp;|&nbsp; <b>Janela:</b> {WINDOW_MINUTES:.1f} min
&nbsp;|&nbsp; <b>Gerado:</b> {datetime.utcnow().isoformat()}Z</p>

<h2>1. Introdução</h2>
<p>Este relatório apresenta um sistema de monitoramento de recursos (CPU, memória e cargas) executado em container Docker,
com aplicação de três modelos de IA: Regressão Linear (previsão), IsolationForest (detecção de anomalias) e KMeans (clusterização).
Os dados são amostrados periodicamente e persistidos em <code>{DB_PATH}</code> (SQLite interno ao container).</p>

<h2>2. Desenvolvimento</h2>
<ul>
  <li><b>Coleta:</b> <code>psutil</code> captura CPU%, Mem% e load average (1m/5m/15m) em intervalos configuráveis (<code>INTERVAL_SEC</code>).</li>
  <li><b>Persistência:</b> amostras gravadas em SQLite para consultas por janela (<code>WINDOW_MINUTES</code>).</li>
  <li><b>Gráficos:</b> PNGs gerados via <code>matplotlib</code> (backend headless) e servidos pelo app.</li>
</ul>

<h2>3. Modelos de IA</h2>
<ul>
  <li><b>Previsão (Regressão Linear):</b> usa defasagens (lags) de CPU para prever CPU futura e reporta MAE e R².</li>
  <li><b>Anomalias (IsolationForest):</b> aprende o padrão multivariado e assinala outliers; exibe taxa de anomalia.</li>
  <li><b>Clusters (KMeans):</b> agrupa perfis de carga (ex.: ocioso, carga, crítico) a partir de CPU%×Mem% e loads.</li>
</ul>

<h2>4. Resultados e Gráficos</h2>
<h3>4.1 CPU e Memória (%)</h3>
<img src="/report/img/cpu_mem.png" alt="CPU e Memória">
<h3>4.2 Loads (1m/5m/15m)</h3>
<img src="/report/img/loads.png" alt="Loads">
<h3>4.3 Forecast (Regressão Linear)</h3>
<img src="/report/img/forecast.png" alt="Forecast">
<h3>4.4 Anomalias (IsolationForest)</h3>
<img src="/report/img/anomaly.png" alt="Anomalias">
<h3>4.5 Clusters (KMeans)</h3>
<img src="/report/img/clusters.png" alt="Clusters">

<h2>5. Evidências de Execução</h2>
<ul>
  <li>API saúde: <code>GET /healthz</code></li>
  <li>Metadados: <code>GET /about</code></li>
  <li>Últimas amostras: <code>GET /metrics</code></li>
  <li>Relatório (esta página): <code>GET /report/html</code></li>
</ul>

<h2>6. Conclusão</h2>
<p>O sistema atende aos requisitos da entrega: coleta contínua, aplicação de três modelos de IA, gráficos de monitoramento e relatório detalhado.
Tudo opera em Docker, sem dependência de VM, com imagens publicáveis no Docker Hub.</p>

<hr>
<p class="meta">Dados mantidos em SQLite interno do container: <code>{DB_PATH}</code>. Para regerar gráficos imediatamente: <code>POST /snapshot</code>.</p>
</body></html>""")

def charts_loop():
    while True:
        try:
            generate_charts()
        except Exception:
            # evita queda do gerador por erro eventual
            pass
        time.sleep(60)

# ------------------------- ROTAS -------------------------
@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/about")
def about():
    info = {
        "app": "ia-collector-cloud",
        "version": "2.2",
        "interval_sec": INTERVAL_SEC,
        "db_path": DB_PATH,
        "window_minutes_default": WINDOW_MINUTES,
        "report_dir": REPORT_DIR
    }
    return jsonify(info)

@app.route("/metrics")
def metrics():
    df = load_window_df(WINDOW_MINUTES)
    if df.empty:
        return jsonify({"samples": [], "interval_sec": INTERVAL_SEC})
    out = df.tail(12).to_dict(orient="records")
    return jsonify({"samples": out, "interval_sec": INTERVAL_SEC})

@app.route("/ai/forecast")
def ai_forecast():
    wm = float(request.args.get("window_minutes", WINDOW_MINUTES))
    lags = int(request.args.get("lags", 5))
    df = load_window_df(wm)
    if df.empty:
        return jsonify({"error": "Dados insuficientes"}), 400
    fc = make_forecast(df.copy(), lags=lags)
    if not fc:
        return jsonify({"error": "Dados insuficientes"}), 400
    return jsonify({
        "window_minutes": wm,
        "lags": lags,
        "n_samples": int(len(fc["df"])),
        "metrics": {"MAE": fc["mae"], "R2": fc["r2"]}
    })

@app.route("/ai/anomaly")
def ai_anomaly():
    wm = float(request.args.get("window_minutes", WINDOW_MINUTES))
    contamination = float(request.args.get("contamination", 0.03))
    df = load_window_df(wm)
    if df.empty:
        return jsonify({"error": "Dados insuficientes"}), 400
    an = make_anomaly(df.copy(), contamination=contamination)
    if not an:
        return jsonify({"error": "Dados insuficientes"}), 400
    return jsonify({
        "window_minutes": wm,
        "contamination": contamination,
        "anom_rate": an["rate"]
    })

@app.route("/ai/clusters")
def ai_clusters():
    wm = float(request.args.get("window_minutes", WINDOW_MINUTES))
    k = int(request.args.get("k", 3))
    df = load_window_df(wm)
    if df.empty:
        return jsonify({"error": "Dados insuficientes"}), 400
    cl = make_clusters(df.copy(), k=k)
    if not cl:
        return jsonify({"error": "Dados insuficientes"}), 400
    return jsonify({
        "window_minutes": wm,
        "k": k,
        "cluster_sizes": cl["sizes"]
    })

@app.route("/report/html")
def report_html():
    return send_from_directory(REPORT_DIR, "index.html")

@app.route("/report/img/<name>")
def report_img(name):
    return send_from_directory(REPORT_DIR, name)

@app.route("/snapshot", methods=["POST", "GET"])
def snapshot():
    generate_charts()
    return jsonify({"ok": True, "generated_at": datetime.utcnow().isoformat()})

@app.route("/dashboard")
def dashboard():
    df = load_window_df(WINDOW_MINUTES)
    n = 0 if df.empty else len(df)
    last = {} if df.empty else df.tail(1).to_dict(orient="records")[0]
    html = f"""
    <html><head><title>IA Collector Cloud</title></head><body>
    <h1>IA Collector Cloud (v2.2)</h1>
    <p><b>Intervalo:</b> {INTERVAL_SEC}s | <b>Janela padrão:</b> {WINDOW_MINUTES} min | <b>Amostras:</b> {n}</p>
    <p><b>Última amostra:</b> {last}</p>
    <h2>Relatório</h2>
    <ul>
      <li><a href="/report/html">/report/html</a></li>
      <li><a href="/metrics">/metrics</a></li>
      <li><a href="/ai/forecast">/ai/forecast</a></li>
      <li><a href="/ai/anomaly">/ai/anomaly</a></li>
      <li><a href="/ai/clusters">/ai/clusters</a></li>
      <li><a href="/about">/about</a></li>
      <li><a href="/healthz">/healthz</a></li>
    </ul>
    </body></html>
    """
    return Response(html, mimetype="text/html")

# ------------------------- STARTUP (Flask 3/Gunicorn) -------------------------
_started = False
_started_lock = threading.Lock()

def _start_background_once():
    global _started
    with _started_lock:
        if not _started:
            init_db()
            seed_demo_if_needed()
            threading.Thread(target=sample_forever, daemon=True).start()
            threading.Thread(target=charts_loop, daemon=True).start()
            _started = True

_start_background_once()

if __name__ == "__main__":
    _start_background_once()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
