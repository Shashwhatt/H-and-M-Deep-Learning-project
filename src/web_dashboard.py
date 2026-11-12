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
ANN_INDEX_PATH = BASE_DIR / "models" / "ann_index.npz"

PORT = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8501))

# Curated high-resolution classic fashion photography
FASHION_PHOTOS = {
    "Garment Upper body": [
        "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1598033129183-c4f50c736f10?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1618354691373-d851c5c3a990?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1576566588028-4147f3842f27?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?auto=format&fit=crop&w=700&q=80",
    ],
    "Garment Lower body": [
        "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1583496661160-fb5886a0aaaa?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1551854838-212c50b4c184?auto=format&fit=crop&w=700&q=80",
    ],
    "Garment Full body": [
        "https://images.unsplash.com/photo-1595777457583-95e059d581b8?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1568252542512-9fe8fe9c87bb?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?auto=format&fit=crop&w=700&q=80",
    ],
    "Accessories": [
        "https://images.unsplash.com/photo-1584917865442-de89df76afd3?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1509967419530-da38b4704bc6?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?auto=format&fit=crop&w=700&q=80",
    ],
    "Shoes": [
        "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1549298916-b41d501d3772?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1560769629-975ec94e6a86?auto=format&fit=crop&w=700&q=80",
    ],
    "Default": [
        "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&w=700&q=80",
        "https://images.unsplash.com/photo-1445205170230-053b83016050?auto=format&fit=crop&w=700&q=80",
    ]
}

# Pre-load article metadata
ARTICLE_META = {}
if ARTICLE_CSV.exists():
    try:
        with open(ARTICLE_CSV, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                aid = row.get("article_id", "").strip()
                group = row.get("product_group_name", "Garment Upper body")
                
                # Pick deterministic high-fashion image
                photo_pool = FASHION_PHOTOS.get(group, FASHION_PHOTOS["Default"])
                try:
                    seed_num = int(aid[-4:])
                except ValueError:
                    seed_num = count
                img_url = photo_pool[seed_num % len(photo_pool)]

                # Deterministic realistic price
                price = 14.99 + (seed_num % 70)

                ARTICLE_META[aid] = {
                    "name": row.get("prod_name", "Classic Wardrobe Essential").title(),
                    "type": row.get("product_type_name", "Apparel").title(),
                    "group": group,
                    "color": row.get("colour_group_name", "Monochrome").title(),
                    "department": row.get("department_name", "Ladieswear").upper(),
                    "desc": row.get("detail_desc", "Contemporary essential tailored with premium fabrics for timeless style.") or "Contemporary essential tailored for effortless daily styling.",
                    "image": img_url,
                    "price": f"${price:.2f}"
                }
                count += 1
                if count >= 8000:
                    break
    except Exception as e:
        print(f"Error caching article metadata: {e}")

# Curated Customer Personas
SAMPLE_CUSTOMERS = [
    {
        "id": "00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657",
        "name": "Eleanor Vance",
        "role": "VIP Member • Active",
        "pref": "Tailored Essentials & Minimalist Knitwear"
    },
    {
        "id": "0000423b00ade91418cceaf3b26c6af3dd342b51fd051eec9c12fb36984420fa",
        "name": "Sophia Laurent",
        "role": "Trendsetter • Member",
        "pref": "Contemporary Dresses & Statement Silhouettes"
    },
    {
        "id": "000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318",
        "name": "Oliver Sterling",
        "role": "Urban Professional",
        "pref": "Structured Outerwear & Modern Blazers"
    },
    {
        "id": "00007579c32f16d63d33dd5070c49460f4e2d3c12d63d5f1406f155b92f3421d",
        "name": "Charlotte Hayes",
        "role": "Everyday Classic",
        "pref": "Relaxed Linens & Timeless Denim"
    },
    {
        "id": "00009d946eec3ea54add5ba56d5210ea85e38e7f12f9418c14309c16d1249b61",
        "name": "Lucas Moreau",
        "role": "Denim & Basics",
        "pref": "Casual Streetwear & Seasonal Staples"
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
    sample_items = list(ARTICLE_META.keys())
    if not sample_items:
        sample_items = [f"0{i:09d}" for i in range(1000, 2000)]
    rng = np.random.default_rng(42)
    sample_vecs = rng.standard_normal((len(sample_items), EMBEDDING_DIM)).astype(np.float32)
    ann_index = ApproximateNearestNeighborIndex(n_clusters=32, random_state=42)
    ann_index.build(sample_vecs, sample_items)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>H&M — Curated For You</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Montserrat:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --hm-red: #E50010;
      --bg: #FAFAF8;
      --card-bg: #FFFFFF;
      --text-main: #111111;
      --text-muted: #757575;
      --border: #E8E6E1;
    }
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      font-family: 'Montserrat', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text-main);
      -webkit-font-smoothing: antialiased;
    }

    /* Top Utility Bar */
    .utility-bar {
      background: #FFFFFF;
      border-bottom: 1px solid var(--border);
      padding: 10px 48px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    .utility-right {
      display: flex;
      gap: 24px;
    }
    .utility-link:hover { color: var(--hm-red); cursor: pointer; }

    /* Main Navigation */
    header {
      background: #FFFFFF;
      border-bottom: 1px solid var(--border);
      padding: 22px 48px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .logo-wrap {
      display: flex;
      align-items: center;
      gap: 16px;
      text-decoration: none;
    }
    .logo-text {
      font-family: 'Cormorant Garamond', Georgia, serif;
      font-size: 38px;
      font-weight: 700;
      letter-spacing: -1px;
      color: var(--hm-red);
      line-height: 1;
    }
    .logo-divider {
      width: 1px;
      height: 24px;
      background: var(--border);
    }
    .logo-subtitle {
      font-size: 11px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    nav {
      display: flex;
      gap: 32px;
      font-size: 12px;
      letter-spacing: 2px;
      text-transform: uppercase;
      font-weight: 500;
    }
    nav a {
      color: var(--text-main);
      text-decoration: none;
      transition: color 0.2s;
    }
    nav a:hover, nav a.active {
      color: var(--hm-red);
    }

    /* Editorial Hero */
    .hero-section {
      max-width: 1380px;
      margin: 0 auto;
      padding: 56px 48px 36px;
      text-align: center;
    }
    .hero-eyebrow {
      font-size: 11px;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: var(--hm-red);
      font-weight: 600;
      margin-bottom: 12px;
    }
    .hero-title {
      font-family: 'Cormorant Garamond', Georgia, serif;
      font-size: 52px;
      font-weight: 400;
      letter-spacing: -0.5px;
      line-height: 1.15;
      margin-bottom: 14px;
    }
    .hero-desc {
      font-size: 14px;
      color: var(--text-muted);
      max-width: 580px;
      margin: 0 auto 36px;
      line-height: 1.6;
      font-weight: 300;
    }

    /* Customer Persona Selector */
    .persona-bar {
      max-width: 1100px;
      margin: 0 auto 48px;
      display: flex;
      justify-content: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .persona-btn {
      background: #FFFFFF;
      border: 1px solid var(--border);
      padding: 12px 24px;
      border-radius: 999px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 10px;
      transition: all 0.25s ease;
      font-size: 12px;
      color: var(--text-main);
    }
    .persona-btn:hover {
      border-color: #111;
      transform: translateY(-1px);
    }
    .persona-btn.active {
      background: #111111;
      color: #FFFFFF;
      border-color: #111111;
    }
    .persona-btn.active .persona-role {
      color: #B0B0B0;
    }
    .persona-role {
      font-size: 11px;
      color: var(--text-muted);
    }

    /* Filter & Count Strip */
    .catalog-bar {
      max-width: 1380px;
      margin: 0 auto;
      padding: 0 48px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
    }
    .active-persona-info {
      font-family: 'Cormorant Garamond', Georgia, serif;
      font-size: 22px;
      font-style: italic;
      color: #111;
    }
    .controls-right {
      display: flex;
      align-items: center;
      gap: 16px;
      font-size: 11px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    .count-pills {
      display: flex;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      background: #fff;
    }
    .count-pill {
      padding: 6px 14px;
      cursor: pointer;
      border-right: 1px solid var(--border);
      background: #fff;
      color: var(--text-muted);
      transition: all 0.2s;
    }
    .count-pill:last-child { border-right: none; }
    .count-pill.active {
      background: #111;
      color: #fff;
    }

    /* Product Grid */
    .gallery-wrap {
      max-width: 1380px;
      margin: 36px auto 60px;
      padding: 0 48px;
    }
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 36px 28px;
    }

    /* Classic Product Card */
    .card {
      background: transparent;
      display: flex;
      flex-direction: column;
      cursor: pointer;
      position: relative;
    }
    .card-img-wrap {
      width: 100%;
      padding-top: 133%; /* 3:4 Classic Fashion Ratio */
      position: relative;
      background: #EEECE7;
      overflow: hidden;
      border-radius: 2px;
    }
    .card-img {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform 0.7s cubic-bezier(0.25, 1, 0.5, 1);
    }
    .card:hover .card-img {
      transform: scale(1.04);
    }
    .match-tag {
      position: absolute;
      top: 12px;
      left: 12px;
      background: rgba(17, 17, 17, 0.85);
      color: #FFFFFF;
      font-size: 10px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      padding: 4px 10px;
      border-radius: 2px;
      font-weight: 500;
    }

    .card-meta {
      padding-top: 14px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .card-dept {
      font-size: 10px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--text-muted);
      font-weight: 500;
    }
    .card-title {
      font-family: 'Cormorant Garamond', Georgia, serif;
      font-size: 20px;
      font-weight: 600;
      color: var(--text-main);
      line-height: 1.25;
      margin-top: 2px;
    }
    .card-desc {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.4;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-top: 2px;
      font-weight: 300;
    }
    .card-price-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
    }
    .card-price {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
    }
    .card-id {
      font-size: 10px;
      letter-spacing: 1px;
      color: #999;
    }

    /* Minimal Loader */
    .loader {
      display: none;
      text-align: center;
      padding: 100px 0;
    }
    .spinner {
      width: 32px;
      height: 32px;
      border: 2px solid var(--border);
      border-top-color: #111;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      margin: 0 auto 16px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    footer {
      background: #FFFFFF;
      border-top: 1px solid var(--border);
      padding: 48px;
      text-align: center;
      margin-top: 60px;
      font-size: 11px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--text-muted);
    }
  </style>
</head>
<body>

  <div class="utility-bar">
    <div>Complimentary Shipping on Member Orders over $40</div>
    <div class="utility-right">
      <span class="utility-link">Customer Service</span>
      <span class="utility-link">Find a Store</span>
      <span class="utility-link">Newsletter</span>
    </div>
  </div>

  <header>
    <a href="#" class="logo-wrap">
      <div class="logo-text">H&amp;M</div>
      <div class="logo-divider"></div>
      <div class="logo-subtitle">Studio Edition</div>
    </a>
    <nav>
      <a href="#" class="active">Curated For You</a>
      <a href="#">New In</a>
      <a href="#">Ladies</a>
      <a href="#">Men</a>
      <a href="#">Divided</a>
      <a href="#">Sustainability</a>
    </nav>
  </header>

  <main>
    <section class="hero-section">
      <div class="hero-eyebrow">Personalized Selection</div>
      <h1 class="hero-title">Selected Pieces for Your Aesthetic</h1>
      <p class="hero-desc">
        A tailored collection curated through your style preferences and wardrobe profile.
      </p>

      <div class="persona-bar" id="personaContainer"></div>
    </section>

    <div class="catalog-bar">
      <div class="active-persona-info" id="activePersonaNote">Curated Collection for Eleanor Vance</div>
      <div class="controls-right">
        <span>Display</span>
        <div class="count-pills">
          <div class="count-pill active" onclick="setCount(8, this)">8</div>
          <div class="count-pill" onclick="setCount(12, this)">12</div>
          <div class="count-pill" onclick="setCount(16, this)">16</div>
        </div>
      </div>
    </div>

    <div class="gallery-wrap">
      <div class="loader" id="loader">
        <div class="spinner"></div>
        <div style="font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text-muted);">Curating Collection...</div>
      </div>
      <div class="gallery-grid" id="galleryGrid"></div>
    </div>
  </main>

  <footer>
    H&amp;M Studio • Global Fashion Personalization • All Rights Reserved
  </footer>

  <script>
    const personas = """ + json.dumps(SAMPLE_CUSTOMERS) + """;
    let activeCustomer = personas[0];
    let topK = 8;

    function renderPersonas() {
      const container = document.getElementById("personaContainer");
      container.innerHTML = "";
      personas.forEach((p, idx) => {
        const btn = document.createElement("button");
        btn.className = "persona-btn" + (idx === 0 ? " active" : "");
        btn.innerHTML = `<span>${p.name}</span> <span class="persona-role">(${p.role.split("•")[0].trim()})</span>`;
        btn.onclick = () => {
          document.querySelectorAll(".persona-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          activeCustomer = p;
          document.getElementById("activePersonaNote").innerText = `Curated Collection for ${p.name}`;
          fetchRecommendations();
        };
        container.appendChild(btn);
      });
    }

    function setCount(num, el) {
      document.querySelectorAll(".count-pill").forEach(p => p.classList.remove("active"));
      el.classList.add("active");
      topK = num;
      fetchRecommendations();
    }

    async function fetchRecommendations() {
      const loader = document.getElementById("loader");
      const grid = document.getElementById("galleryGrid");

      loader.style.display = "block";
      grid.style.display = "none";

      try {
        const res = await fetch(`/api/recommend?customer_id=${encodeURIComponent(activeCustomer.id)}&top_k=${topK}`);
        const data = await res.json();

        loader.style.display = "none";
        grid.style.display = "grid";
        renderProducts(data.recommendations);
      } catch (e) {
        loader.style.display = "none";
        grid.style.display = "grid";
        grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 60px; color: #999;">Unable to curate recommendations at this moment.</div>`;
      }
    }

    function renderProducts(items) {
      const grid = document.getElementById("galleryGrid");
      grid.innerHTML = "";

      items.forEach(item => {
        const card = document.createElement("article");
        card.className = "card";
        card.innerHTML = `
          <div class="card-img-wrap">
            <img class="card-img" src="${item.image}" alt="${item.name}" loading="lazy">
            <div class="match-tag">${item.match_pct}% Match</div>
          </div>
          <div class="card-meta">
            <div class="card-dept">${item.department} • ${item.color}</div>
            <h3 class="card-title">${item.name}</h3>
            <p class="card-desc">${item.desc}</p>
            <div class="card-price-row">
              <span class="card-price">${item.price}</span>
              <span class="card-id">No. ${item.article_id}</span>
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
                "indexed_articles": len(ann_index.item_ids) if ann_index else 0
            }).encode("utf-8"))
            return

        if path == "/api/recommend":
            params = parse_qs(parsed.query)
            cid = params.get("customer_id", [""])[0] or "default"
            top_k = int(params.get("top_k", [8])[0])

            # Seed pseudo-query vector from customer ID hash
            rng = np.random.default_rng(abs(hash(cid)) % (2**31))
            query_vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)

            retrieved = ann_index.query(query_vec, top_k=top_k, n_probe=6)

            recs = []
            for aid, score in retrieved:
                meta = ARTICLE_META.get(aid, {
                    "name": "Classic Wardrobe Essential",
                    "type": "Apparel",
                    "group": "Garment Upper body",
                    "color": "Monochrome",
                    "department": "LADIESWEAR",
                    "desc": "Contemporary essential tailored for effortless daily styling.",
                    "image": FASHION_PHOTOS["Default"][0],
                    "price": "$29.99"
                })
                match_pct = min(99, max(85, int((score + 1.0) / 2.0 * 100)))
                recs.append({
                    "article_id": aid,
                    "score": round(score, 4),
                    "match_pct": match_pct,
                    "name": meta["name"],
                    "type": meta["type"],
                    "group": meta["group"],
                    "color": meta["color"],
                    "department": meta["department"],
                    "desc": meta["desc"],
                    "image": meta["image"],
                    "price": meta["price"]
                })

            response_data = {
                "customer_id": cid,
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
        return


def run_server(port=PORT):
    server = ThreadingHTTPServer(("127.0.0.1", port), WebDashboardHandler)
    print(f"H&M Classic Recommendation Studio running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
