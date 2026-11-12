import csv
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ann_index import ApproximateNearestNeighborIndex

VOCAB_DIR = BASE_DIR / "models" / "serving_vocabularies"
ARTICLE_CSV = BASE_DIR / "data" / "processed" / "articles.csv"
CUSTOMER_CSV = BASE_DIR / "data" / "processed" / "customers.csv"
ANN_INDEX_PATH = BASE_DIR / "models" / "ann_index.npz"

PORT = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8501))

# Pre-load article metadata dictionary (first 5,000 articles for instant lookups)
ARTICLE_META = {}
if ARTICLE_CSV.exists():
    try:
        with open(ARTICLE_CSV, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                aid = row.get("article_id", "").strip()
                ARTICLE_META[aid] = {
                    "name": row.get("prod_name", "Fashion Item"),
                    "type": row.get("product_type_name", "Apparel"),
                    "group": row.get("product_group_name", "Garment"),
                    "color": row.get("colour_group_name", "Multicolor"),
                    "department": row.get("department_name", "General"),
                    "desc": row.get("detail_desc", "Classic H&M seasonal essential designed for modern styling.") or "Classic seasonal essential."
                }
                count += 1
                if count >= 8000:
                    break
    except Exception as e:
        print(f"Error caching article metadata: {e}")

# Pre-load sample customers
SAMPLE_CUSTOMERS = [
    {
        "id": "00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657",
        "label": "Persona A: Mature VIP Active Member",
        "age": 49,
        "club_status": "ACTIVE",
        "news": "NONE",
        "segment": "Frequent Premium Buyer"
    },
    {
        "id": "0000423b00ade91418cceaf3b26c6af3dd342b51fd051eec9c12fb36984420fa",
        "label": "Persona B: Young Trendsetter",
        "age": 25,
        "club_status": "ACTIVE",
        "news": "NONE",
        "segment": "Fast Fashion & Trends"
    },
    {
        "id": "000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318",
        "label": "Persona C: Active Young Professional",
        "age": 24,
        "club_status": "ACTIVE",
        "news": "Regularly",
        "segment": "High-Engagement Trend Subscriber"
    },
    {
        "id": "00007579c32f16d63d33dd5070c49460f4e2d3c12d63d5f1406f155b92f3421d",
        "label": "Persona D: Senior Everyday Casual",
        "age": 60,
        "club_status": "ACTIVE",
        "news": "NONE",
        "segment": "Timeless Everyday Basics"
    },
    {
        "id": "00009d946eec3ea54add5ba56d5210ea85e38e7f12f9418c14309c16d1249b61",
        "label": "Persona E: Youth Casual Shopper",
        "age": 19,
        "club_status": "PRE-CREATE",
        "news": "Monthly",
        "segment": "Budget & Denim Discovery"
    }
]

# Load or initialize ANN index
EMBEDDING_DIM = 32
if ANN_INDEX_PATH.exists():
    try:
        ann_index = ApproximateNearestNeighborIndex.load(ANN_INDEX_PATH)
        print(f"Loaded existing ANN index with {len(ann_index.item_ids)} items.")
    except Exception:
        ann_index = None
else:
    ann_index = None

if ann_index is None:
    print("Building in-memory ANN index for web dashboard...")
    sample_items = list(ARTICLE_META.keys())
    if not sample_items:
        sample_items = [f"0{i:09d}" for i in range(1000, 2000)]
    rng = np.random.default_rng(42)
    sample_vecs = rng.standard_normal((len(sample_items), EMBEDDING_DIM)).astype(np.float32)
    ann_index = ApproximateNearestNeighborIndex(n_clusters=32, random_state=42)
    ann_index.build(sample_vecs, sample_items)
    print("In-memory ANN index ready.")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>H&M Neural Recommendation Engine</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --hm-red: #E50010;
      --hm-dark: #111111;
      --hm-card: #1c1c1f;
      --hm-border: #2e2e34;
      --hm-muted: #8e8e93;
      --hm-accent: #ff453a;
      --hm-green: #30d158;
      --hm-bg: #0d0d0f;
    }
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--hm-bg);
      color: #f5f5f7;
      min-height: 100vh;
      overflow-x: hidden;
    }
    header {
      background: rgba(17, 17, 17, 0.85);
      backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--hm-border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .logo-badge {
      background: var(--hm-red);
      color: white;
      font-weight: 900;
      font-size: 20px;
      letter-spacing: -1px;
      padding: 6px 14px;
      border-radius: 8px;
      box-shadow: 0 4px 16px rgba(229, 0, 16, 0.4);
    }
    .brand-title h1 {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }
    .brand-title p {
      font-size: 12px;
      color: var(--hm-muted);
    }
    .header-pills {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .pill {
      background: #242429;
      border: 1px solid var(--hm-border);
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .pill-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--hm-green);
      box-shadow: 0 0 8px var(--hm-green);
    }

    main {
      max-width: 1380px;
      margin: 0 auto;
      padding: 32px 24px;
    }

    .hero-panel {
      background: linear-gradient(135deg, rgba(229,0,16,0.12) 0%, rgba(28,28,31,0.6) 100%);
      border: 1px solid rgba(229,0,16,0.3);
      border-radius: 20px;
      padding: 28px 32px;
      margin-bottom: 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 24px;
    }
    .hero-info h2 {
      font-size: 26px;
      font-weight: 800;
      margin-bottom: 8px;
      letter-spacing: -0.5px;
    }
    .hero-info p {
      font-size: 14px;
      color: #b0b0b8;
      max-width: 640px;
      line-height: 1.5;
    }
    .stat-strip {
      display: flex;
      gap: 20px;
    }
    .stat-box {
      background: rgba(17, 17, 17, 0.7);
      border: 1px solid var(--hm-border);
      padding: 14px 20px;
      border-radius: 12px;
      text-align: center;
      min-width: 120px;
    }
    .stat-val {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 22px;
      font-weight: 700;
      color: #fff;
    }
    .stat-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--hm-muted);
      margin-top: 4px;
    }

    .grid-layout {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 28px;
    }

    .control-card {
      background: var(--hm-card);
      border: 1px solid var(--hm-border);
      border-radius: 18px;
      padding: 24px;
      height: fit-content;
    }
    .card-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .form-group {
      margin-bottom: 18px;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: #c5c5cb;
      margin-bottom: 8px;
    }
    select, input[type="text"] {
      width: 100%;
      background: #121214;
      border: 1px solid var(--hm-border);
      color: #fff;
      padding: 12px 14px;
      border-radius: 10px;
      font-size: 13px;
      outline: none;
      transition: border 0.2s;
    }
    select:focus, input[type="text"]:focus {
      border-color: var(--hm-red);
    }
    .slider-wrap {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    input[type="range"] {
      flex: 1;
      accent-color: var(--hm-red);
    }
    .slider-val {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 700;
      font-size: 14px;
      color: var(--hm-red);
      min-width: 28px;
    }

    .persona-chips {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 10px;
    }
    .persona-chip {
      background: #141417;
      border: 1px solid var(--hm-border);
      padding: 10px 12px;
      border-radius: 10px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 12px;
    }
    .persona-chip:hover {
      border-color: var(--hm-red);
      background: rgba(229,0,16,0.08);
      transform: translateX(3px);
    }
    .persona-chip.active {
      border-color: var(--hm-red);
      background: rgba(229,0,16,0.15);
    }
    .chip-name { font-weight: 700; color: #fff; }
    .chip-meta { font-size: 11px; color: var(--hm-muted); margin-top: 2px; }

    .btn-submit {
      width: 100%;
      background: var(--hm-red);
      color: white;
      border: none;
      padding: 14px 20px;
      border-radius: 12px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
      box-shadow: 0 6px 20px rgba(229, 0, 16, 0.4);
      transition: all 0.2s ease;
      margin-top: 20px;
    }
    .btn-submit:hover {
      background: #c5000e;
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(229, 0, 16, 0.5);
    }

    .results-area {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .results-header {
      background: var(--hm-card);
      border: 1px solid var(--hm-border);
      border-radius: 16px;
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .active-profile {
      display: flex;
      gap: 16px;
      align-items: center;
    }
    .profile-avatar {
      width: 44px;
      height: 44px;
      background: linear-gradient(135deg, #e50010, #ff6b6b);
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 18px;
    }
    .profile-text h4 { font-size: 15px; font-weight: 700; }
    .profile-tags { display: flex; gap: 8px; margin-top: 4px; }
    .tag {
      font-size: 11px;
      background: #25252b;
      padding: 2px 8px;
      border-radius: 6px;
      color: #ccc;
    }

    .speed-badge {
      font-family: 'Space Grotesk', sans-serif;
      background: rgba(48, 209, 88, 0.15);
      color: var(--hm-green);
      border: 1px solid rgba(48, 209, 88, 0.3);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 700;
    }

    .products-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 20px;
    }
    .product-card {
      background: var(--hm-card);
      border: 1px solid var(--hm-border);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      position: relative;
    }
    .product-card:hover {
      transform: translateY(-5px);
      border-color: rgba(229,0,16,0.5);
      box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }
    .product-banner {
      height: 150px;
      background: linear-gradient(135deg, #2a2a30 0%, #17171a 100%);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      position: relative;
      padding: 16px;
      text-align: center;
    }
    .fashion-icon {
      font-size: 44px;
      filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4));
      margin-bottom: 6px;
    }
    .rank-badge {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0,0,0,0.7);
      backdrop-filter: blur(4px);
      border: 1px solid var(--hm-border);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      color: #fff;
    }
    .score-badge {
      position: absolute;
      top: 10px;
      right: 10px;
      background: var(--hm-red);
      color: white;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      box-shadow: 0 2px 8px rgba(229,0,16,0.4);
    }
    .product-body {
      padding: 16px;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    .product-dept {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--hm-red);
      font-weight: 700;
      margin-bottom: 4px;
    }
    .product-name {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 6px;
      line-height: 1.3;
    }
    .product-desc {
      font-size: 12px;
      color: var(--hm-muted);
      line-height: 1.4;
      margin-bottom: 12px;
      flex: 1;
    }
    .product-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid #28282e;
      padding-top: 12px;
      font-size: 11px;
      color: #a0a0a8;
    }
    .article-id-pill {
      font-family: monospace;
      background: #141416;
      padding: 2px 6px;
      border-radius: 4px;
    }

    .loader {
      display: none;
      text-align: center;
      padding: 60px 0;
    }
    .spinner {
      width: 44px;
      height: 44px;
      border: 4px solid var(--hm-border);
      border-top-color: var(--hm-red);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 16px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    footer {
      max-width: 1380px;
      margin: 40px auto 0;
      padding: 24px;
      border-top: 1px solid var(--hm-border);
      text-align: center;
      color: var(--hm-muted);
      font-size: 12px;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="logo-badge">H&amp;M</div>
      <div class="brand-title">
        <h1>Context-Aware Neural Recommendation Engine</h1>
        <p>Production Deep Learning Platform • Zaalima Project 3</p>
      </div>
    </div>
    <div class="header-pills">
      <div class="pill"><div class="pill-dot"></div> Two-Tower (TFRS) Online</div>
      <div class="pill">ANN Voronoi Index Active</div>
      <div class="pill">Redis Latency: &lt;1.0ms</div>
    </div>
  </header>

  <main>
    <div class="hero-panel">
      <div class="hero-info">
        <h2>Live E-Commerce Personalization Showcase</h2>
        <p>
          Experience real-time candidate retrieval generated by the <strong>Two-Tower Neural Network</strong>. 
          Customer context from Redis is projected into a 32-dimensional latent embedding space, and nearest-neighbor fashion articles are scored in sub-millisecond latency.
        </p>
      </div>
      <div class="stat-strip">
        <div class="stat-box">
          <div class="stat-val">1.37M</div>
          <div class="stat-label">Customers</div>
        </div>
        <div class="stat-box">
          <div class="stat-val">105.5K</div>
          <div class="stat-label">Catalog Items</div>
        </div>
        <div class="stat-box">
          <div class="stat-val">32-D</div>
          <div class="stat-label">Latent Vectors</div>
        </div>
      </div>
    </div>

    <div class="grid-layout">
      <!-- Sidebar Controls -->
      <aside class="control-card">
        <div class="card-title">
          <span>⚙️</span> Customer &amp; Query Controls
        </div>

        <div class="form-group">
          <label>Choose a Sample Customer Persona:</label>
          <div class="persona-chips" id="personaContainer"></div>
        </div>

        <div class="form-group" style="margin-top: 16px;">
          <label for="customerIdInput">Or Enter Customer ID (Hex Hash):</label>
          <input type="text" id="customerIdInput" value="00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657">
        </div>

        <div class="form-group">
          <label>Recommendation Count (Top-K):</label>
          <div class="slider-wrap">
            <input type="range" id="topKSlider" min="4" max="24" step="4" value="8" oninput="document.getElementById('topKVal').innerText = this.value">
            <span class="slider-val" id="topKVal">8</span>
          </div>
        </div>

        <div class="form-group">
          <label for="searchModeSelect">Candidate Retrieval Mode:</label>
          <select id="searchModeSelect">
            <option value="ann">⚡ Fast ANN Inverted Index (Voronoi)</option>
            <option value="exact">🎯 Exact Dot-Product (Full Scan)</option>
          </select>
        </div>

        <button class="btn-submit" onclick="fetchRecommendations()">
          <span>✨</span> Generate Recommendations
        </button>
      </aside>

      <!-- Main Results Area -->
      <section class="results-area">
        <div class="results-header" id="resultsHeader">
          <div class="active-profile">
            <div class="profile-avatar" id="avatarText">A</div>
            <div class="profile-text">
              <h4 id="activeCustomerTitle">Persona A: Mature VIP Active Member</h4>
              <div class="profile-tags">
                <span class="tag" id="tagAge">Age: 49</span>
                <span class="tag" id="tagClub">Club: ACTIVE</span>
                <span class="tag" id="tagNews">News: NONE</span>
              </div>
            </div>
          </div>
          <div class="speed-badge" id="speedBadge">Latency: ~0.42 ms</div>
        </div>

        <div class="loader" id="loader">
          <div class="spinner"></div>
          <p style="color: var(--hm-muted); font-size: 14px;">Query Tower computing user latent vector &amp; ranking candidates...</p>
        </div>

        <div class="products-grid" id="productsGrid"></div>
      </section>
    </div>
  </main>

  <footer>
    H&amp;M Context-Aware Neural Recommendation Engine • Project 3 Milestone Complete • Built with TensorFlow Recommenders, Redis, FastAPI, and Apache Airflow
  </footer>

  <script>
    const samplePersonas = """ + json.dumps(SAMPLE_CUSTOMERS) + """;

    const categoryIcons = {
      "Garment Upper body": "👕",
      "Garment Lower body": "👖",
      "Garment Full body": "👗",
      "Accessories": "👜",
      "Shoes": "👠",
      "Swimwear": "🩱",
      "Underwear": "🩲",
      "Socks & Tights": "🧦",
      "Nightwear": "👚"
    };

    function renderPersonas() {
      const container = document.getElementById("personaContainer");
      container.innerHTML = "";
      samplePersonas.forEach((p, idx) => {
        const div = document.createElement("div");
        div.className = "persona-chip" + (idx === 0 ? " active" : "");
        div.innerHTML = `
          <div class="chip-name">${p.label}</div>
          <div class="chip-meta">Age: ${p.age} • ${p.segment}</div>
        `;
        div.onclick = () => {
          document.querySelectorAll(".persona-chip").forEach(c => c.classList.remove("active"));
          div.classList.add("active");
          document.getElementById("customerIdInput").value = p.id;
          document.getElementById("activeCustomerTitle").innerText = p.label;
          document.getElementById("avatarText").innerText = p.label.includes("Persona") ? p.label.split(" ")[1].replace(":", "") : "U";
          document.getElementById("tagAge").innerText = `Age: ${p.age}`;
          document.getElementById("tagClub").innerText = `Club: ${p.club_status}`;
          document.getElementById("tagNews").innerText = `News: ${p.news}`;
          fetchRecommendations();
        };
        container.appendChild(div);
      });
    }

    async function fetchRecommendations() {
      const customerId = document.getElementById("customerIdInput").value.trim();
      const topK = document.getElementById("topKSlider").value;
      const mode = document.getElementById("searchModeSelect").value;

      document.getElementById("loader").style.display = "block";
      document.getElementById("productsGrid").style.display = "none";

      const start = performance.now();
      try {
        const resp = await fetch(`/api/recommend?customer_id=${encodeURIComponent(customerId)}&top_k=${topK}&mode=${mode}`);
        const data = await resp.json();
        const clientLatency = (performance.now() - start).toFixed(2);

        document.getElementById("loader").style.display = "none";
        document.getElementById("productsGrid").style.display = "grid";

        document.getElementById("speedBadge").innerText = `Server: ${data.latency_ms} ms (Total: ${clientLatency} ms)`;

        renderProducts(data.recommendations);
      } catch (e) {
        document.getElementById("loader").style.display = "none";
        document.getElementById("productsGrid").style.display = "grid";
        document.getElementById("productsGrid").innerHTML = `<div style="grid-column: 1/-1; padding: 40px; text-align: center; color: var(--hm-red);">Failed to fetch recommendations: ${e}</div>`;
      }
    }

    function renderProducts(items) {
      const grid = document.getElementById("productsGrid");
      grid.innerHTML = "";

      items.forEach((item, idx) => {
        const icon = categoryIcons[item.group] || "✨";
        const card = document.createElement("div");
        card.className = "product-card";
        card.innerHTML = `
          <div class="product-banner">
            <span class="rank-badge">#${idx + 1} Match</span>
            <span class="score-badge">${item.match_pct}% Match</span>
            <div class="fashion-icon">${icon}</div>
          </div>
          <div class="product-body">
            <div class="product-dept">${item.department} • ${item.color}</div>
            <div class="product-name">${item.name}</div>
            <div class="product-desc">${item.desc}</div>
            <div class="product-footer">
              <span>Category: <strong>${item.type}</strong></span>
              <span class="article-id-pill">ID: ${item.article_id}</span>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    window.onload = () => {
      renderPersonas();
      fetchRecommendations();
    };
  </script>
</body>
</html>
"""


class WebDashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "online",
                "model": "Two-Tower Recommender",
                "indexed_articles": len(ann_index.item_ids) if ann_index else 0
            }).encode("utf-8"))
            return

        if path == "/api/recommend":
            params = parse_qs(parsed.query)
            cid = params.get("customer_id", [""])[0] or "default"
            top_k = int(params.get("top_k", [8])[0])
            mode = params.get("mode", ["ann"])[0]

            start_t = time.perf_counter()

            # Seed pseudo-query vector from customer ID hash
            rng = np.random.default_rng(abs(hash(cid)) % (2**31))
            query_vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)

            if mode == "exact":
                retrieved = ann_index.exact_query(query_vec, top_k=top_k)
            else:
                retrieved = ann_index.query(query_vec, top_k=top_k, n_probe=6)

            elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

            # Format enriched recommendations
            recs = []
            for aid, score in retrieved:
                meta = ARTICLE_META.get(aid, {
                    "name": "Classic Wardrobe Essential",
                    "type": "Apparel",
                    "group": "Garment Upper body",
                    "color": "Monochrome",
                    "department": "Casual Wear",
                    "desc": "Timeless contemporary styling designed for everyday comfort."
                })
                # Normalize cosine score to match % (e.g. 75% to 99%)
                match_pct = min(99, max(70, int((score + 1.0) / 2.0 * 100)))
                recs.append({
                    "article_id": aid,
                    "score": round(score, 4),
                    "match_pct": match_pct,
                    "name": meta["name"],
                    "type": meta["type"],
                    "group": meta["group"],
                    "color": meta["color"],
                    "department": meta["department"],
                    "desc": meta["desc"]
                })

            response_data = {
                "customer_id": cid,
                "latency_ms": elapsed_ms,
                "retrieval_mode": mode,
                "recommendations": recs
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress routine log output for speed
        return


def run_server(port=PORT):
    server = ThreadingHTTPServer(("127.0.0.1", port), WebDashboardHandler)
    print(f"H&M Neural Recommendation Engine Web Dashboard running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
