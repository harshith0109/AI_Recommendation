# =============================================================
#  app.py  –  Flask Backend for Shop Recommendation System
#  Run:  python app.py
#  Requires:  flask, flask-cors, numpy, pandas, scikit-learn, scipy
#  Install:   pip install flask flask-cors numpy pandas scikit-learn scipy
# =============================================================

import numpy as np
import pandas as pd
import sqlite3
import os
import threading
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

app = Flask(__name__, static_folder=".")
CORS(app)

# =============================================================
#  PATHS
# =============================================================
BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "shop.db")

# =============================================================
#  SVD RETRAINING CONFIG
# =============================================================
RETRAIN_EVERY_N = 10   # retrain SVD after every 10 new DB interactions

# =============================================================
#  DATABASE
# =============================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id     TEXT PRIMARY KEY,
                password    TEXT,
                is_csv_user INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT NOT NULL,
                product_id    TEXT NOT NULL,
                rating        INTEGER NOT NULL DEFAULT 5,
                interacted_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, product_id) ON CONFLICT REPLACE
            );

            -- Track how many DB interactions existed at last SVD retrain
            CREATE TABLE IF NOT EXISTS svd_meta (
                id                  INTEGER PRIMARY KEY CHECK (id = 1),
                last_retrain_count  INTEGER DEFAULT 0,
                last_retrained_at   TEXT
            );
            INSERT OR IGNORE INTO svd_meta(id, last_retrain_count) VALUES (1, 0);

            CREATE INDEX IF NOT EXISTS idx_inter_user    ON interactions(user_id);
            CREATE INDEX IF NOT EXISTS idx_inter_product ON interactions(product_id);
        """)
    print("[db] SQLite ready →", DB_PATH)


def seed_csv_users():
    with get_db() as conn:
        already = conn.execute(
            "SELECT COUNT(*) FROM users WHERE is_csv_user=1"
        ).fetchone()[0]
        if already > 0:
            print(f"[db] {already} CSV users already seeded, skipping.")
            return
        csv_users = [(uid, None, 1) for uid in svd_state["user_item_matrix"].index.tolist()]
        conn.executemany(
            "INSERT OR IGNORE INTO users(user_id, password, is_csv_user) VALUES (?,?,?)",
            csv_users
        )
        print(f"[db] Seeded {len(csv_users)} CSV users.")


# ── DB helpers ──────────────────────────────────────────────

def get_user_interactions(user_id: str) -> list:
    # Pull DB interactions (new purchases made in the app)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT product_id FROM interactions WHERE user_id=? ORDER BY interacted_at ASC",
            (user_id,)
        ).fetchall()
    db_pids = [r["product_id"] for r in rows]

    # Also include CSV history for existing users
    csv_pids = df_interactions[df_interactions["user_id"] == user_id]["product_id"].tolist()

    # Merge: CSV first (historical), DB at the end (most recent)
    # so user_products[-1] always = the last thing they actually bought
    seen = set()
    merged = []
    for pid in csv_pids:
        if pid not in seen:
            seen.add(pid)
            merged.append(pid)
    for pid in db_pids:
        if pid in seen:
            merged = [p for p in merged if p != pid]  # remove from old position
        seen.add(pid)
        merged.append(pid)  # always at the end

    return merged


def get_user_interaction_count(user_id: str) -> int:
    return len(get_user_interactions(user_id))


def get_total_db_interaction_count() -> int:
    with get_db() as conn:
        return conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]


def get_last_retrain_count() -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT last_retrain_count FROM svd_meta WHERE id=1"
        ).fetchone()[0]


def update_retrain_count(count: int):
    with get_db() as conn:
        conn.execute(
            "UPDATE svd_meta SET last_retrain_count=?, last_retrained_at=? WHERE id=1",
            (count, datetime.utcnow().isoformat())
        )


def save_interaction(user_id: str, product_id: str, rating: int):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO interactions(user_id, product_id, rating, interacted_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, product_id) DO UPDATE SET
                   rating        = excluded.rating,
                   interacted_at = excluded.interacted_at""",
            (user_id, product_id, rating, datetime.utcnow().isoformat())
        )


def get_all_db_interactions() -> pd.DataFrame:
    """Pull every interaction from SQLite as a DataFrame."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT user_id, product_id, rating FROM interactions"
        ).fetchall()
    return pd.DataFrame(rows, columns=["user_id", "product_id", "rating"])

# =============================================================
#  LOAD CSV DATA  (fast — just reading files, safe at import time)
# =============================================================
df_interactions = pd.read_csv(os.path.join(BASE, "users_interactions.csv"))
df_products     = pd.read_csv(os.path.join(BASE, "products_catalog.csv"))
df_interactions = df_interactions.drop(columns=["timestamp"], errors="ignore")

print(f"[data] {len(df_interactions)} interactions  |  {len(df_products)} products")

existing_users = set(df_interactions["user_id"].unique())

# =============================================================
#  SVD STATE  –  lives in a dict so retrain can swap atomically
# =============================================================
svd_lock  = threading.Lock()
svd_state = {}   # populated lazily on first request

# Popularity tables – also populated lazily
product_pop        = None
product_popularity = None


def build_svd_model(interactions_df: pd.DataFrame):
    """
    Build mean-centered SVD from a combined interactions DataFrame
    (CSV + any DB interactions merged together).
    Returns a dict with all matrices needed for recommendations.
    """
    # Merge CSV + DB, DB rating wins on conflict (it's more recent)
    merged = (
        interactions_df
        .sort_values("interacted_at" if "interacted_at" in interactions_df.columns else "user_id")
        .drop_duplicates(subset=["user_id", "product_id"], keep="last")
    )

    user_item_matrix = merged.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        aggfunc="mean"
    ).fillna(0)

    sparse = csr_matrix(user_item_matrix.values)

    # Per-user mean (vectorized)
    user_means = np.zeros(sparse.shape[0])
    for i in range(sparse.shape[0]):
        row_data = sparse[i].data
        if len(row_data) > 0:
            user_means[i] = row_data.mean()

    # Mean-center
    mc = sparse.tocoo().astype(float)
    mc.data -= user_means[mc.row]
    mc = mc.tocsr()

    k = min(50, min(mc.shape) - 1)
    U, sigma, Vt = svds(mc, k=k)

    predicted = np.dot(np.dot(U, np.diag(sigma)), Vt)
    predicted = predicted + user_means.reshape(-1, 1)
    predicted = np.clip(predicted, 1, 5)

    print(f"[SVD] retrained — shape: {predicted.shape}  k={k}  "
          f"users: {len(user_item_matrix)}  interactions: {len(merged)}")

    return {
        "user_item_matrix":  user_item_matrix,
        "sparse_matrix":     sparse,
        "predicted_ratings": predicted,
        "user_means":        user_means,
    }


def retrain_svd_if_needed():
    """
    Called after every purchase. If RETRAIN_EVERY_N new DB interactions
    have accumulated since the last retrain, trigger a background retrain.
    """
    total = get_total_db_interaction_count()
    last  = get_last_retrain_count()

    if total - last < RETRAIN_EVERY_N:
        return   # not enough new data yet

    # Fire retrain in background so the HTTP response isn't blocked
    threading.Thread(target=_do_retrain, daemon=True).start()


def _do_retrain():
    """Background thread: merge CSV + DB, rebuild SVD, swap state atomically."""
    try:
        print("[SVD] retraining started...")

        db_df  = get_all_db_interactions()
        db_df["interacted_at"] = "9999"

        csv_df = df_interactions.copy()
        csv_df["interacted_at"] = "0000"

        combined  = pd.concat([csv_df, db_df], ignore_index=True)
        new_state = build_svd_model(combined)

        with svd_lock:
            svd_state.update(new_state)

        update_retrain_count(get_total_db_interaction_count())
        print("[SVD] retrain complete ✓")

    except Exception as e:
        print(f"[SVD] retrain failed: {e}")


# =============================================================
#  LAZY STARTUP
#  All heavy work (SVD build, popularity tables, DB init) is
#  deferred until the first real HTTP request so that gunicorn
#  can bind to the port and satisfy Render's health-check
#  within the 30-second timeout window.
# =============================================================
_startup_done = False
_startup_lock = threading.Lock()


def _run_startup():
    global _startup_done, product_pop, product_popularity

    with _startup_lock:
        if _startup_done:
            return
        _startup_done = True

    print("[startup] building models…")

    # 1. SVD (the slow part)
    csv_with_ts = df_interactions.copy()
    csv_with_ts["interacted_at"] = "0000"
    svd_state.update(build_svd_model(csv_with_ts))

    # 2. Category popularity tables
    _pp = (
        df_interactions.groupby("product_id")
        .size()
        .reset_index(name="interaction_count")
    )
    _pp = _pp.merge(df_products[["product_id", "category"]], on="product_id", how="left")
    _pp["rank_in_category"] = (
        _pp.groupby("category")["interaction_count"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    product_pop        = _pp.sort_values(["category", "rank_in_category"])
    product_popularity = df_interactions["product_id"].value_counts()

    # 3. DB init + seed
    init_db()
    seed_csv_users()

    print("[startup] all models ready ✓")


@app.before_request
def lazy_startup():
    if not _startup_done:
        _run_startup()


# =============================================================
#  RECOMMENDATION HELPERS
# =============================================================

def products_to_list(df_slice: pd.DataFrame) -> list:
    cols = ["product_id", "product_name", "brand", "category",
            "price", "description", "image_url"]
    return df_slice[cols].to_dict(orient="records")


def recommend_popular(top_n: int = 12) -> list:
    pids = product_popularity.head(top_n).index
    return products_to_list(df_products[df_products["product_id"].isin(pids)])


def get_random_3_from_last_category(user_product_ids: list, exclude_pids: set) -> list:
    """Pick 3 random products from the category of the last purchased item."""
    if not user_product_ids:
        return []
    last_pid = user_product_ids[-1]
    category = get_category_for_product(last_pid)
    if not category:
        return []
    pool = df_products[
        (df_products["category"] == category) &
        (~df_products["product_id"].isin(exclude_pids))
    ]
    sample = pool.sample(min(3, len(pool)), random_state=None)
    return products_to_list(sample)


# ── SVD recommendation ──────────────────────────────────────

def recommend_svd(user_id: str, top_n: int = 12, exclude_pids: list = None) -> list:
    """
    Uses the current svd_state (always up to date after retrains).
    Works for both CSV users and new users if they appear in the
    retrained matrix.
    """
    with svd_lock:
        uim       = svd_state["user_item_matrix"]
        sparse    = svd_state["sparse_matrix"]
        predicted = svd_state["predicted_ratings"]

    if user_id not in uim.index:
        return recommend_popular(top_n)

    user_index = uim.index.get_loc(user_id)
    user_preds = predicted[user_index].copy()
    prod_ids   = uim.columns

    # Mask already-rated items (from matrix)
    already = sparse[user_index].toarray().flatten()
    user_preds[already > 0] = -np.inf

    # Mask additionally excluded products (recent DB purchases)
    if exclude_pids:
        for pid in exclude_pids:
            if pid in prod_ids:
                user_preds[prod_ids.get_loc(pid)] = -np.inf

    rec_indices = np.argsort(user_preds)[-9:][::-1]
    rec_pids    = prod_ids[rec_indices]
    svd_recs    = products_to_list(df_products[df_products["product_id"].isin(rec_pids)])

    actual_user_products = get_user_interactions(user_id)
    exclude_cat = set(exclude_pids or []) | set(rec_pids)
    cat3 = get_random_3_from_last_category(actual_user_products, exclude_cat)
    cat3_ids = {p["product_id"] for p in cat3}
    combined = cat3 + [p for p in svd_recs if p["product_id"] not in cat3_ids]
    return combined[:12]


# ── Find most similar CSV user (proxy SVD for new users) ────

def find_most_similar_csv_user(user_product_ids: list):
    with svd_lock:
        uim = svd_state["user_item_matrix"]

    vec = np.zeros(uim.shape[1])
    for pid in user_product_ids:
        if pid in uim.columns:
            idx    = uim.columns.get_loc(pid)
            mean_r = df_interactions[df_interactions["product_id"] == pid]["rating"].mean()
            vec[idx] = mean_r if not np.isnan(mean_r) else 3.0

    if vec.sum() == 0:
        return None

    sims     = cosine_similarity(vec.reshape(1, -1), uim.values).flatten()
    best_idx = sims.argmax()
    best_sim = sims[best_idx]

    if best_sim < 0.1:
        return None

    return uim.index[best_idx]


# ── Session-Aware Popularity Recommender (1–4 purchases) ────

SLOT_ALLOCATION = {
    1: [12],
    2: [6, 6],
    3: [6, 3, 3],
    4: [6, 2, 2, 2],
    5: [4, 2, 2, 2, 2],
}


def get_category_for_product(product_id: str):
    match = df_products.loc[df_products["product_id"] == product_id, "category"]
    return match.values[0] if not match.empty else None


def get_top_products_in_category(category: str, n: int, exclude: set) -> list:
    pool = product_pop[
        (product_pop["category"] == category) &
        (~product_pop["product_id"].isin(exclude))
    ]
    return pool.nsmallest(n, "rank_in_category")["product_id"].tolist()


def recommend_session_popularity(user_product_ids: list, top_n: int = 12) -> list:
    if not user_product_ids:
        return []

    window = list(user_product_ids[-5:])
    slots  = SLOT_ALLOCATION[len(window)]

    viewed      = set(user_product_ids)
    recommended = set()
    result_pids = []

    for slot_idx, slot_count in enumerate(slots):
        product_id = window[-(slot_idx + 1)]
        category   = get_category_for_product(product_id)
        if not category:
            continue

        exclude = viewed | recommended
        top     = get_top_products_in_category(category, slot_count, exclude)
        result_pids.extend(top)
        recommended.update(top)

    if not result_pids:
        return recommend_popular(top_n)

    order_map = {pid: i for i, pid in enumerate(result_pids)}
    rec_df    = df_products[df_products["product_id"].isin(result_pids)].copy()
    rec_df["_order"] = rec_df["product_id"].map(order_map)
    return products_to_list(rec_df.sort_values("_order").drop(columns="_order"))


# ── User-based CF (5–14 purchases) ──────────────────────────

def recommend_user_based(user_product_ids: list, top_n: int = 12) -> list:
    if not user_product_ids:
        return recommend_popular(top_n)

    with svd_lock:
        uim = svd_state["user_item_matrix"]

    vec = np.zeros(uim.shape[1])
    for pid in user_product_ids:
        if pid in uim.columns:
            idx    = uim.columns.get_loc(pid)
            mean_r = df_interactions[df_interactions["product_id"] == pid]["rating"].mean()
            vec[idx] = mean_r if not np.isnan(mean_r) else 3.0

    sims          = cosine_similarity(vec.reshape(1, -1), uim.values).flatten()
    top_indices   = sims.argsort()[-10:][::-1]
    similar_users = uim.index[top_indices]

    similar_data = df_interactions[df_interactions["user_id"].isin(similar_users)]
    counts       = similar_data["product_id"].value_counts()
    counts       = counts.drop(labels=user_product_ids, errors="ignore")
    rec_pids  = counts.head(9).index
    ub_recs   = products_to_list(df_products[df_products["product_id"].isin(rec_pids)])

    exclude_cat = set(rec_pids) | set(user_product_ids)
    cat3 = get_random_3_from_last_category(user_product_ids, exclude_cat)
    cat3_ids = {p["product_id"] for p in cat3}
    combined = cat3 + [p for p in ub_recs if p["product_id"] not in cat3_ids]
    return combined[:12]


# =============================================================
#  MASTER ROUTING LOGIC
# =============================================================

def get_recommendations_for_user(user_id: str, top_n: int = 12) -> dict:
    """
    Decision tree (fresh on every call):
      0 purchases              → popularity
      1–4 purchases            → session-aware popularity (slot allocation)
      5–14 purchases           → user-based CF
      15+ purchases, CSV user  → SVD (retrained matrix includes their new purchases)
      15+ purchases, new user  → SVD of most similar CSV user (proxy)
                                 fallback: user-based CF if no good proxy found
    """
    user_products = get_user_interactions(user_id)
    count         = len(user_products)

    if count == 0:
        return {
            "recommendations": recommend_popular(top_n),
            "method": "popular",
            "history_count": 0,
            "slot_allocation": []
        }

    if count <= 4:
        window = user_products[-5:]
        slots  = SLOT_ALLOCATION[len(window)]
        recs   = recommend_session_popularity(user_products, top_n)
        return {
            "recommendations": recs,
            "method": "session_popularity",
            "history_count": count,
            "slot_allocation": [
                {"product_id": pid, "slots": s}
                for pid, s in zip(reversed(window), slots)
            ]
        }

    if count <= 14:
        return {
            "recommendations": recommend_user_based(user_products, top_n),
            "method": "user_based",
            "history_count": count,
            "slot_allocation": []
        }

    # 15+ purchases
    if user_id in existing_users:
        # CSV user — retrained SVD now includes their DB purchases
        return {
            "recommendations": recommend_svd(user_id, top_n, exclude_pids=user_products),
            "method": "svd",
            "history_count": count,
            "slot_allocation": []
        }
    else:
        # New user — find most similar CSV user, use their SVD predictions
        similar_csv_user = find_most_similar_csv_user(user_products)
        if similar_csv_user:
            return {
                "recommendations": recommend_svd(
                    similar_csv_user, top_n, exclude_pids=user_products
                ),
                "method": "svd_proxy",
                "proxy_user": similar_csv_user,
                "history_count": count,
                "slot_allocation": []
            }
        return {
            "recommendations": recommend_user_based(user_products, top_n),
            "method": "user_based",
            "history_count": count,
            "slot_allocation": []
        }


# =============================================================
#  API ROUTES
# =============================================================

@app.route("/")
def serve_index():
    return send_from_directory(".", "shop.html")


# ── AUTH ────────────────────────────────────────────────────

@app.route("/api/signup", methods=["POST"])
def signup():
    data     = request.json or {}
    user_id  = (data.get("user_id") or "").strip().upper()
    password = (data.get("password") or "").strip()

    if not user_id or not password:
        return jsonify({"error": "user_id and password are required"}), 400

    with get_db() as conn:
        existing = conn.execute(
            "SELECT user_id, is_csv_user FROM users WHERE user_id=?", (user_id,)
        ).fetchone()

    if existing:
        msg = "User ID already exists. Please log in." if existing["is_csv_user"] \
              else "User ID already taken. Choose another."
        return jsonify({"error": msg}), 409

    with get_db() as conn:
        conn.execute(
            "INSERT INTO users(user_id, password, is_csv_user) VALUES (?,?,0)",
            (user_id, password)
        )
    return jsonify({"ok": True, "user_id": user_id, "is_new": True})


@app.route("/api/login", methods=["POST"])
def login():
    data     = request.json or {}
    user_id  = (data.get("user_id") or "").strip().upper()
    password = (data.get("password") or "").strip()

    if not user_id or not password:
        return jsonify({"error": "user_id and password are required"}), 400

    with get_db() as conn:
        user = conn.execute(
            "SELECT user_id, password, is_csv_user FROM users WHERE user_id=?", (user_id,)
        ).fetchone()

    if not user:
        return jsonify({"error": "User not found. Please sign up first."}), 404

    if not user["is_csv_user"] and user["password"] != password:
        return jsonify({"error": "Incorrect password"}), 401

    return jsonify({
        "ok": True,
        "user_id": user_id,
        "is_new": not bool(user["is_csv_user"]),
        "history_count": get_user_interaction_count(user_id)
    })


# ── RECOMMENDATIONS ─────────────────────────────────────────

@app.route("/api/recommendations", methods=["GET"])
def recommendations():
    user_id = (request.args.get("user_id", "") or "").strip().upper() or None
    top_n   = int(request.args.get("top_n", 12))

    if not user_id:
        return jsonify({
            "recommendations": recommend_popular(top_n),
            "method": "popular",
            "history_count": 0,
            "slot_allocation": []
        })

    return jsonify(get_recommendations_for_user(user_id, top_n))


# ── PRODUCTS ────────────────────────────────────────────────

@app.route("/api/products", methods=["GET"])
def products():
    category = request.args.get("category", "").strip()
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))

    subset = df_products
    if category:
        subset = subset[subset["category"].str.lower() == category.lower()]

    total = len(subset)
    start = (page - 1) * per_page
    return jsonify({
        "products": products_to_list(subset.iloc[start:start + per_page]),
        "total":    total,
        "page":     page,
        "per_page": per_page
    })


@app.route("/api/categories", methods=["GET"])
def categories():
    return jsonify({"categories": df_products["category"].unique().tolist()})


# ── INTERACT ────────────────────────────────────────────────

@app.route("/api/interact", methods=["POST"])
def interact():
    data       = request.json or {}
    user_id    = (data.get("user_id") or "").strip().upper()
    product_id = (data.get("product_id") or "").strip()
    rating     = int(data.get("rating", 5))

    if not user_id or not product_id:
        return jsonify({"error": "user_id and product_id are required"}), 400

    with get_db() as conn:
        user = conn.execute(
            "SELECT user_id FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
    if not user:
        return jsonify({"error": "User not found. Please log in first."}), 404

    if product_id not in df_products["product_id"].values:
        return jsonify({"error": f"Product '{product_id}' not found."}), 404

    save_interaction(user_id, product_id, rating)

    # Check if SVD needs retraining (runs in background, non-blocking)
    retrain_svd_if_needed()

    new_count = get_user_interaction_count(user_id)
    new_recs  = get_recommendations_for_user(user_id, top_n=12)

    return jsonify({
        "ok": True,
        "interactions": new_count,
        "updated_recommendations": new_recs
    })


# ── HISTORY ─────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def history():
    user_id = (request.args.get("user_id", "") or "").strip().upper()
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    with get_db() as conn:
        rows = conn.execute(
            "SELECT product_id, rating, interacted_at FROM interactions "
            "WHERE user_id=? ORDER BY interacted_at ASC",
            (user_id,)
        ).fetchall()

    pid_info = df_products.set_index("product_id")[
        ["product_name", "brand", "category", "price"]
    ].to_dict(orient="index")

    history_list = []
    for r in rows:
        info = pid_info.get(r["product_id"], {})
        history_list.append({
            "product_id":    r["product_id"],
            "rating":        r["rating"],
            "interacted_at": r["interacted_at"],
            "product_name":  info.get("product_name", "Unknown"),
            "brand":         info.get("brand", ""),
            "category":      info.get("category", ""),
            "price":         info.get("price", 0),
        })

    return jsonify({
        "user_id": user_id,
        "count":   len(history_list),
        "history": history_list
    })


# ── SVD STATUS (debug endpoint) ─────────────────────────────

@app.route("/api/svd_status", methods=["GET"])
def svd_status():
    with get_db() as conn:
        meta = conn.execute(
            "SELECT last_retrain_count, last_retrained_at FROM svd_meta WHERE id=1"
        ).fetchone()
    total = get_total_db_interaction_count()
    return jsonify({
        "total_db_interactions":      total,
        "last_retrain_count":         meta["last_retrain_count"],
        "last_retrained_at":          meta["last_retrained_at"],
        "interactions_since_retrain": total - meta["last_retrain_count"],
        "retrain_every_n":            RETRAIN_EVERY_N,
        "next_retrain_in":            RETRAIN_EVERY_N - (total - meta["last_retrain_count"])
    })


# =============================================================
#  MAIN
# =============================================================
if __name__ == "__main__":
    print("\n  Shop backend running →  http://localhost:5000\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)