# Shop — Recommendation System Web App

## Folder structure (put everything together)
```
your-folder/
├── app.py
├── shop.html
├── users_interactions.csv
└── products_catalog.csv
```

## 1. Install dependencies
```bash
pip install flask flask-cors numpy pandas scikit-learn scipy
```

## 2. Run the backend
```bash
python app.py
```
You'll see: `Shop backend running → http://localhost:5000`  
SVD model trains automatically at startup (~5–10 seconds).

## 3. Open the frontend
Open `shop.html` directly in your browser  
**OR** visit `http://localhost:5000` (Flask serves it too).

---

## How it works

| Visitor type | Purchases | Recommendation method |
|---|---|---|
| Guest (not logged in) | — | Popularity-based (most interacted products) |
| Any user | 0 | Popularity-based |
| Any user | 1–4 | Session-aware popularity (slot allocation) |
| Any user | 5–14 | User-based Collaborative Filtering |
| Existing CSV user (U0001–U1000) | 15+ | SVD — Matrix Factorisation (retrained with new purchases) |
| New sign-up | 15+ | SVD of most similar CSV user (proxy) |
| New sign-up (no good proxy found) | 15+ | User-based Collaborative Filtering |

The method label shows live in the **"Recommended for you"** section.

---

## Session-aware slot allocation (1–4 purchases)

Recommendations are split across the last 5 viewed product categories, with more slots given to the most recently purchased item. Sliding window of 5 — oldest drops off when a 6th product is added.

| Products purchased | Slot split (out of 12) |
|---|---|
| 1 | 12 from current |
| 2 | 6 current + 6 prev1 |
| 3 | 6 current + 3 prev1 + 3 prev2 |
| 4 | 6 current + 2 prev1 + 2 prev2 + 2 prev3 |
| 5+ | 4 current + 2 prev1 + 2 prev2 + 2 prev3 + 2 prev4 |

---

## SVD retraining

The SVD model retrains automatically in the **background** every **10 new purchases** (configurable via `RETRAIN_EVERY_N` in `app.py`). Retraining merges the original CSV interactions with all new DB purchases so existing users' recommendations evolve as they buy more.

Check retrain status at any time:
```
GET http://localhost:5000/api/svd_status
```

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/login` | POST | Login existing or new user |
| `/api/signup` | POST | Register a new user |
| `/api/recommendations?user_id=U0001` | GET | Get personalised recommendations |
| `/api/products?category=Electronics` | GET | Get products by category |
| `/api/categories` | GET | List all categories |
| `/api/interact` | POST | Record a purchase / rating |
| `/api/history?user_id=U0001` | GET | Get a user's full purchase history |
| `/api/svd_status` | GET | Check SVD retrain status |

---

## Notes
- **Existing users (U0001–U1000):** any password accepted — no passwords stored in CSV data.
- **New sign-up users:** password stored in SQLite (`shop.db`) — persists across restarts.
- **Ratings** recorded via the Purchased tab feed back into recommendations in real time.
- **SVD retraining** runs in a background thread — API responses are never blocked.
- Change `RETRAIN_EVERY_N = 10` at the top of `app.py` to retrain more or less frequently.
