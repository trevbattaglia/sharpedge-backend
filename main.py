from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

app = FastAPI(title="SharpEdge Actions", version="1.0.0")

# --- Helpers (math) ---
def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return (-odds) / (-odds + 100)

def decimal_from_american(odds: int) -> float:
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / -odds)

def devig_two_way(p1: float, p2: float) -> (float, float):
    s = p1 + p2
    return (p1 / s, p2 / s) if s > 0 else (0.0, 0.0)

def ev_per_dollar(q: float, odds: int) -> float:
    b = decimal_from_american(odds) - 1.0
    return q * b - (1 - q)

def kelly_half(q: float, odds: int) -> float:
    b = decimal_from_american(odds) - 1.0
    k = (b*q - (1-q)) / b if b > 0 else 0.0
    return max(0.0, k/2)  # half‑Kelly, floor at 0

# --- Endpoints (stubs you can extend) ---

@app.get("/odds")
def get_odds(sport: str, markets: Optional[List[str]] = None, books: Optional[List[str]] = None):
    """
    Returns stub multi‑book odds for one MLB game so Actions wiring works.
    """
    data = {
        "sport": sport,
        "markets": markets or ["ml", "total"],
        "books": books or ["dk", "mgm", "czu"],
        "games": [
            {
                "game_id": "mlb-nyy-hou-2025-08-11",
                "away": "HOU",
                "home": "NYY",
                "markets": {
                    "ml": {
                        "HOU": {"dk": +110, "mgm": +112, "czu": +115},
                        "NYY": {"dk": -130, "mgm": -128, "czu": -125}
                    },
                    "total": {
                        "8.5_over": {"dk": -105, "mgm": -110},
                        "8.5_under": {"dk": -115, "mgm": -110}
                    }
                }
            }
        ]
    }
    return JSONResponse(data)

@app.get("/mlb/stats")
def mlb_stats(date: Optional[str] = None):
    return JSONResponse({
        "date": date,
        "games": [
            {
                "game_id": "mlb-nyy-hou-2025-08-11",
                "probables": {"HOU": "Framber Valdez", "NYY": "Carlos Rodon"},
                "bullpen_rest_index": {"HOU": 0.62, "NYY": 0.48},  # 0 (gassed) → 1 (fresh)
                "park": "Yankee Stadium"
            }
        ]
    })

@app.get("/mlb/savant")
def mlb_savant(player_id: str):
    # toy example: pitch mix and simple platoon split
    return JSONResponse({
        "player_id": player_id,
        "pitch_mix": {"sinker": 42.0, "curve": 27.0, "change": 12.0, "cutter": 11.0, "four_seam": 8.0},
        "splits": {"vs_L": {"wOBA": 0.305}, "vs_R": {"wOBA": 0.320}}
    })

@app.get("/fangraphs/projections")
def fangraphs_projections(entity_id: str):
    return JSONResponse({
        "entity_id": entity_id,
        "projections": {
            "K%": 24.5, "BB%": 7.1, "ERA": 3.65, "wOBA": 0.334, "TB_per_game": 1.75
        }
    })

@app.post("/model/probability")
def model_probability(payload: Dict[str, Any] = Body(...)):
    """
    Accepts anything; returns a sample model win prob for a two‑way ML.
    payload suggestion:
      {"game_id":"mlb-nyy-hou-2025-08-11","sides":{"HOU":{}, "NYY":{}}}
    """
    q = {"HOU": 0.47, "NYY": 0.53}
    return JSONResponse({"q": q, "meta": {"note": "stubbed probs"}})

@app.post("/rank")
def rank(payload: Dict[str, Any] = Body(...)):
    """
    Input (example):
    {
      "market":"ml",
      "game_id":"mlb-nyy-hou-2025-08-11",
      "sides":{"HOU":+115,"NYY":-125},
      "model_q":{"HOU":0.47,"NYY":0.53}
    }
    """
    market = payload.get("market", "ml")
    sides = payload.get("sides", {})
    model_q = payload.get("model_q", {})
    keys = list(sides.keys())
    if len(keys) != 2:
        return JSONResponse({"error": "only two‑way example in stub"}, status_code=400)

    a, b = keys[0], keys[1]
    p_a = american_to_implied(int(sides[a]))
    p_b = american_to_implied(int(sides[b]))
    fair_a, fair_b = devig_two_way(p_a, p_b)

    rows = []
    for team, fair in [(a, fair_a), (b, fair_b)]:
        q = float(model_q.get(team, fair))  # fallback q=fair
        odds = int(sides[team])
        edge = q - fair
        ev = ev_per_dollar(q, odds)
        k = kelly_half(q, odds)
        rows.append({
            "Market": market.upper(),
            "Side": team,
            "Line": odds,
            "Fair%": round(fair*100, 1),
            "Model%": round(q*100, 1),
            "Edge": round(edge*100, 1),
            "EV_per_$": round(ev, 4),
            "Kelly_half": round(k, 3),
            "Notes": "stub calc"
        })
    # sort by EV desc
    rows.sort(key=lambda r: r["EV_per_$"], reverse=True)
    return JSONResponse({"cards": rows})

@app.get("/news/consensus")
def news(sport: str):
    return JSONResponse({
        "sport": sport,
        "consensus": [],
        "injuries": [],
        "links": []
    })
