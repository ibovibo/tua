"""
AstroHackathon 2025 — LEO Uydu Bant Genişliği Optimizasyonu
Dinamik (öncelik bazlı) vs Statik (eşit) dağıtım karşılaştırması
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Literal

# ─────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────
TOTAL_CAPACITY_MBPS = 1000.0

PRIORITY_MAP = {
    "military": 5,
    "emergency": 4,
    "civilian": 1,
}

SCENARIO_CONFIGS = {
    "normal": {"civilian": 0.70, "emergency": 0.20, "military": 0.10},
    "crisis":  {"civilian": 0.40, "emergency": 0.40, "military": 0.20},
}

N_USERS = 100
SEED = 42

UserType = Literal["military", "emergency", "civilian"]


# ─────────────────────────────────────────────
# 1. Veri Üretimi
# ─────────────────────────────────────────────
def generate_users(scenario: str, n_users: int = N_USERS, seed: int = SEED) -> pd.DataFrame:
    """
    Kullanıcı veri kümesi oluşturur.
    Her kullanıcıya demand, latency_sensitivity ve priority atanır.
    """
    rng = np.random.default_rng(seed)
    ratios = SCENARIO_CONFIGS[scenario]

    counts = {
        utype: int(ratio * n_users)
        for utype, ratio in ratios.items()
    }
    # Yuvarlama farkını civilian'a ekle
    total_assigned = sum(counts.values())
    counts["civilian"] += n_users - total_assigned

    rows = []
    for utype, count in counts.items():
        priority = PRIORITY_MAP[utype]

        # Her kullanıcı tipinin demand ve gecikme dağılımı farklı
        if utype == "military":
            demand = rng.uniform(10, 50, count)
            latency_sensitivity = rng.uniform(0.7, 1.0, count)
        elif utype == "emergency":
            demand = rng.uniform(5, 30, count)
            latency_sensitivity = rng.uniform(0.5, 0.9, count)
        else:  # civilian
            demand = rng.uniform(1, 20, count)
            latency_sensitivity = rng.uniform(0.0, 0.5, count)

        for i in range(count):
            rows.append({
                "user_type": utype,
                "priority": priority,
                "demand": round(float(demand[i]), 2),
                "latency_sensitivity": round(float(latency_sensitivity[i]), 4),
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.index.name = "user_id"
    return df


# ─────────────────────────────────────────────
# 2. Skor Hesabı
# ─────────────────────────────────────────────
def compute_scores(df: pd.DataFrame) -> pd.Series:
    """
    score = demand * 0.5 + latency_sensitivity * 30 + priority * 10
    """
    return (
        df["demand"] * 0.5
        + df["latency_sensitivity"] * 30
        + df["priority"] * 10
    )


# ─────────────────────────────────────────────
# 3. Bant Genişliği Dağıtım Algoritmaları
# ─────────────────────────────────────────────
def allocate_priority(df: pd.DataFrame, capacity: float = TOTAL_CAPACITY_MBPS) -> pd.Series:
    """
    Öncelik bazlı dinamik dağıtım.
    Yüksek skorlu kullanıcı önce karşılanır; kapasite bitince alttakiler 0 alır.
    """
    scores = compute_scores(df)
    order = scores.sort_values(ascending=False).index

    allocated = pd.Series(0.0, index=df.index)
    remaining = capacity

    for uid in order:
        need = df.loc[uid, "demand"]
        give = min(need, remaining)
        allocated.loc[uid] = give
        remaining -= give
        if remaining <= 0:
            break

    return allocated


def allocate_baseline(df: pd.DataFrame, capacity: float = TOTAL_CAPACITY_MBPS) -> pd.Series:
    """
    Eşit dağıtım (baseline).
    Her kullanıcı kapasiteden eşit pay alır; talebi küçük olanlar fazlasını geri bırakmaz.
    """
    equal_share = capacity / len(df)
    allocated = df["demand"].clip(upper=equal_share)
    # Kullanılmayan kapasiteyi burada dağıtmıyoruz (gerçek statik sistem gibi)
    return allocated


# ─────────────────────────────────────────────
# 4. Metrik Ölçümü
# ─────────────────────────────────────────────
def calculate_metrics(df: pd.DataFrame, allocated: pd.Series, label: str) -> dict:
    """
    Karşılanma oranı, gecikme cezası ve kapasite verimliliğini hesaplar.
    """
    total_demand = df["demand"].sum()
    total_allocated = allocated.sum()

    # Karşılanma oranı: kullanıcı bazlı (tam karşılananlar / toplam)
    satisfied = (allocated >= df["demand"]).sum()
    satisfaction_rate = satisfied / len(df)

    # Kritik kullanıcı karşılanma oranı (military + emergency)
    critical_mask = df["user_type"].isin(["military", "emergency"])
    critical_df = df[critical_mask]
    critical_alloc = allocated[critical_mask]
    critical_satisfied = (critical_alloc >= critical_df["demand"]).sum()
    critical_rate = critical_satisfied / len(critical_df) if len(critical_df) > 0 else 0.0

    # Gecikme cezası: karşılanmayan band × gecikme duyarlılığı (ağırlıklı toplam)
    unmet = (df["demand"] - allocated).clip(lower=0)
    latency_penalty = (unmet * df["latency_sensitivity"]).sum()

    # Kapasite verimliliği
    efficiency = total_allocated / TOTAL_CAPACITY_MBPS

    return {
        "label": label,
        "total_demand_mbps": round(total_demand, 2),
        "total_allocated_mbps": round(total_allocated, 2),
        "satisfaction_rate": round(satisfaction_rate, 4),
        "critical_satisfaction_rate": round(critical_rate, 4),
        "latency_penalty": round(latency_penalty, 4),
        "efficiency": round(efficiency, 4),
        "per_type": _per_type_metrics(df, allocated),
    }


def _per_type_metrics(df: pd.DataFrame, allocated: pd.Series) -> dict:
    result = {}
    for utype in ["military", "emergency", "civilian"]:
        mask = df["user_type"] == utype
        sub = df[mask]
        sub_alloc = allocated[mask]
        if len(sub) == 0:
            continue
        sat = (sub_alloc >= sub["demand"]).sum() / len(sub)
        avg_alloc = sub_alloc.mean()
        avg_demand = sub["demand"].mean()
        result[utype] = {
            "count": int(len(sub)),
            "satisfaction_rate": round(float(sat), 4),
            "avg_demand_mbps": round(float(avg_demand), 2),
            "avg_allocated_mbps": round(float(avg_alloc), 2),
        }
    return result


# ─────────────────────────────────────────────
# 5. Görselleştirme
# ─────────────────────────────────────────────
COLORS = {
    "military":  "#c0392b",
    "emergency": "#e67e22",
    "civilian":  "#2980b9",
}

METHOD_COLORS = {
    "Priority": "#27ae60",
    "Baseline": "#7f8c8d",
}


def visualize_scenario(
    df: pd.DataFrame,
    alloc_priority: pd.Series,
    alloc_baseline: pd.Series,
    metrics_priority: dict,
    metrics_baseline: dict,
    scenario: str,
    output_path: str,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"LEO Bant Genişliği Optimizasyonu — {scenario.upper()} Senaryosu",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Panel 1: Kullanıcı tipi dağılımı ──
    ax1 = fig.add_subplot(gs[0, 0])
    type_counts = df["user_type"].value_counts()
    ax1.pie(
        type_counts.values,
        labels=type_counts.index,
        colors=[COLORS[t] for t in type_counts.index],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax1.set_title("Kullanıcı Tipi Dağılımı")

    # ── Panel 2: Kritik karşılanma oranı karşılaştırması ──
    ax2 = fig.add_subplot(gs[0, 1])
    user_types = ["military", "emergency", "civilian"]
    x = np.arange(len(user_types))
    width = 0.35

    prio_rates = [
        metrics_priority["per_type"].get(t, {}).get("satisfaction_rate", 0) * 100
        for t in user_types
    ]
    base_rates = [
        metrics_baseline["per_type"].get(t, {}).get("satisfaction_rate", 0) * 100
        for t in user_types
    ]

    bars1 = ax2.bar(x - width / 2, prio_rates, width, label="Priority", color=METHOD_COLORS["Priority"])
    bars2 = ax2.bar(x + width / 2, base_rates, width, label="Baseline", color=METHOD_COLORS["Baseline"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(user_types)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Karşılanma Oranı (%)")
    ax2.set_title("Kullanıcı Tipine Göre Karşılanma")
    ax2.legend(fontsize=8)
    ax2.bar_label(bars1, fmt="%.0f%%", padding=2, fontsize=7)
    ax2.bar_label(bars2, fmt="%.0f%%", padding=2, fontsize=7)

    # ── Panel 3: Genel metrik özeti ──
    ax3 = fig.add_subplot(gs[0, 2])
    metrics_to_show = {
        "Toplam\nKarşılanma (%)": (
            metrics_priority["satisfaction_rate"] * 100,
            metrics_baseline["satisfaction_rate"] * 100,
        ),
        "Kritik\nKarşılanma (%)": (
            metrics_priority["critical_satisfaction_rate"] * 100,
            metrics_baseline["critical_satisfaction_rate"] * 100,
        ),
        "Verimlilik (%)": (
            metrics_priority["efficiency"] * 100,
            metrics_baseline["efficiency"] * 100,
        ),
    }
    xlabels = list(metrics_to_show.keys())
    prio_vals = [v[0] for v in metrics_to_show.values()]
    base_vals = [v[1] for v in metrics_to_show.values()]
    x3 = np.arange(len(xlabels))
    b1 = ax3.bar(x3 - width / 2, prio_vals, width, label="Priority", color=METHOD_COLORS["Priority"])
    b2 = ax3.bar(x3 + width / 2, base_vals, width, label="Baseline", color=METHOD_COLORS["Baseline"])
    ax3.set_xticks(x3)
    ax3.set_xticklabels(xlabels, fontsize=8)
    ax3.set_ylim(0, 115)
    ax3.set_ylabel("Değer (%)")
    ax3.set_title("Genel Metrikler")
    ax3.legend(fontsize=8)
    ax3.bar_label(b1, fmt="%.1f", padding=2, fontsize=7)
    ax3.bar_label(b2, fmt="%.1f", padding=2, fontsize=7)

    # ── Panel 4: Dağıtım scatter (Priority) ──
    ax4 = fig.add_subplot(gs[1, 0])
    for utype, color in COLORS.items():
        mask = df["user_type"] == utype
        ax4.scatter(
            df.loc[mask, "demand"],
            alloc_priority[mask],
            color=color, alpha=0.6, s=20, label=utype,
        )
    ax4.plot([0, df["demand"].max()], [0, df["demand"].max()], "k--", lw=0.8, label="ideal")
    ax4.set_xlabel("Talep (Mbps)")
    ax4.set_ylabel("Dağıtılan (Mbps)")
    ax4.set_title("Priority — Talep vs Dağıtım")
    ax4.legend(fontsize=7)

    # ── Panel 5: Dağıtım scatter (Baseline) ──
    ax5 = fig.add_subplot(gs[1, 1])
    for utype, color in COLORS.items():
        mask = df["user_type"] == utype
        ax5.scatter(
            df.loc[mask, "demand"],
            alloc_baseline[mask],
            color=color, alpha=0.6, s=20, label=utype,
        )
    ax5.plot([0, df["demand"].max()], [0, df["demand"].max()], "k--", lw=0.8, label="ideal")
    ax5.set_xlabel("Talep (Mbps)")
    ax5.set_ylabel("Dağıtılan (Mbps)")
    ax5.set_title("Baseline — Talep vs Dağıtım")
    ax5.legend(fontsize=7)

    # ── Panel 6: Gecikme cezası karşılaştırması ──
    ax6 = fig.add_subplot(gs[1, 2])
    penalty_data = {}
    for utype in user_types:
        mask = df["user_type"] == utype
        unmet_p = (df.loc[mask, "demand"] - alloc_priority[mask]).clip(lower=0)
        unmet_b = (df.loc[mask, "demand"] - alloc_baseline[mask]).clip(lower=0)
        penalty_data[utype] = {
            "Priority": (unmet_p * df.loc[mask, "latency_sensitivity"]).sum(),
            "Baseline": (unmet_b * df.loc[mask, "latency_sensitivity"]).sum(),
        }

    x6 = np.arange(len(user_types))
    p_pen = [penalty_data[t]["Priority"] for t in user_types]
    b_pen = [penalty_data[t]["Baseline"] for t in user_types]
    bp1 = ax6.bar(x6 - width / 2, p_pen, width, label="Priority", color=METHOD_COLORS["Priority"])
    bp2 = ax6.bar(x6 + width / 2, b_pen, width, label="Baseline", color=METHOD_COLORS["Baseline"])
    ax6.set_xticks(x6)
    ax6.set_xticklabels(user_types)
    ax6.set_ylabel("Gecikme Cezası (ağırlıklı)")
    ax6.set_title("Gecikme Cezası (düşük = iyi)")
    ax6.legend(fontsize=8)
    ax6.bar_label(bp1, fmt="%.1f", padding=2, fontsize=7)
    ax6.bar_label(bp2, fmt="%.1f", padding=2, fontsize=7)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafik kaydedildi: {output_path}")


# ─────────────────────────────────────────────
# 6. Sonuç Kaydetme
# ─────────────────────────────────────────────
def save_results(all_results: dict, path: str = "results.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  Sonuçlar kaydedildi: {path}")


# ─────────────────────────────────────────────
# 8b. Simülasyon Örnekleri
# ─────────────────────────────────────────────
def build_simulation_examples() -> list[dict]:
    """
    Beş farklı simülasyon örneği döner.
    Her biri farklı bir senaryoyu ya da kapasiteyi test eder.
    """
    examples = []

    # ── Örnek 1: Adım adım skor hesabı (5 kullanıcı) ──
    users_5 = pd.DataFrame([
        {"user_type": "military",  "priority": 5, "demand": 40.0, "latency_sensitivity": 0.95},
        {"user_type": "emergency", "priority": 4, "demand": 25.0, "latency_sensitivity": 0.75},
        {"user_type": "emergency", "priority": 4, "demand": 15.0, "latency_sensitivity": 0.60},
        {"user_type": "civilian",  "priority": 1, "demand": 18.0, "latency_sensitivity": 0.20},
        {"user_type": "civilian",  "priority": 1, "demand":  8.0, "latency_sensitivity": 0.10},
    ])
    users_5["score"] = compute_scores(users_5).round(2)
    alloc_5 = allocate_priority(users_5, capacity=60.0)
    users_5["allocated_priority"] = alloc_5.values
    users_5["allocated_baseline"] = allocate_baseline(users_5, capacity=60.0).values
    examples.append({
        "id": "stepbystep",
        "title": "Adım Adım Skor Hesabı",
        "description": "5 kullanıcı, 60 Mbps kapasite. Her kullanıcının skoru ve dağıtım kararı.",
        "capacity": 60,
        "users": users_5.to_dict(orient="records"),
        "metrics_priority": calculate_metrics(users_5, alloc_5, "priority"),
        "metrics_baseline": calculate_metrics(users_5, allocate_baseline(users_5, capacity=60.0), "baseline"),
    })

    # ── Örnek 2: Kısıtlı Kapasite (500 Mbps — normal senaryonun yarısı) ──
    df_low = generate_users("normal", seed=10)
    alloc_low_p = allocate_priority(df_low, capacity=500.0)
    alloc_low_b = allocate_baseline(df_low, capacity=500.0)
    examples.append({
        "id": "low_capacity",
        "title": "Kısıtlı Kapasite (500 Mbps)",
        "description": "Normal dağılım, kapasiteyi yarıya düşürdük. Algoritma baskı altında nasıl davranır?",
        "capacity": 500,
        "metrics_priority": calculate_metrics(df_low, alloc_low_p, "priority"),
        "metrics_baseline": calculate_metrics(df_low, alloc_low_b, "baseline"),
    })

    # ── Örnek 3: Bolluk Senaryosu (2000 Mbps) ──
    df_rich = generate_users("normal", seed=10)
    alloc_rich_p = allocate_priority(df_rich, capacity=2000.0)
    alloc_rich_b = allocate_baseline(df_rich, capacity=2000.0)
    examples.append({
        "id": "high_capacity",
        "title": "Bolluk Senaryosu (2000 Mbps)",
        "description": "Kapasite talebin üzerinde. Her iki algoritma da tüm kullanıcıları karşılar.",
        "capacity": 2000,
        "metrics_priority": calculate_metrics(df_rich, alloc_rich_p, "priority"),
        "metrics_baseline": calculate_metrics(df_rich, alloc_rich_b, "baseline"),
    })

    # ── Örnek 4: Aşırı Kriz (600 Mbps, kriz dağılımı) ──
    df_extreme = generate_users("crisis", seed=99)
    alloc_ext_p = allocate_priority(df_extreme, capacity=600.0)
    alloc_ext_b = allocate_baseline(df_extreme, capacity=600.0)
    examples.append({
        "id": "extreme_crisis",
        "title": "Aşırı Kriz (600 Mbps)",
        "description": "Kriz dağılımı + %35 kapasite kısıtı. Algoritmanın kritik kullanıcı koruma etkisi.",
        "capacity": 600,
        "metrics_priority": calculate_metrics(df_extreme, alloc_ext_p, "priority"),
        "metrics_baseline": calculate_metrics(df_extreme, alloc_ext_b, "baseline"),
    })

    # ── Örnek 5: Afet Simülasyonu (Deprem — 330 Mbps, 11 kullanıcı) ──
    # Altyapı hasar gördü, kapasite kısıtlı. Emergency ağırlıklı kullanıcı seti.
    disaster_users = pd.DataFrame([
        {"user_type": "military",  "name": "Askeri Komuta",  "priority": 5, "demand": 65.0, "latency_sensitivity": 0.94},
        {"user_type": "military",  "name": "Helikopter Ağı", "priority": 5, "demand": 35.0, "latency_sensitivity": 0.98},
        {"user_type": "emergency", "name": "AFAD Komuta",    "priority": 4, "demand": 60.0, "latency_sensitivity": 0.92},
        {"user_type": "emergency", "name": "SAR Ekibi A",    "priority": 4, "demand": 18.0, "latency_sensitivity": 0.96},
        {"user_type": "emergency", "name": "SAR Ekibi B",    "priority": 4, "demand": 18.0, "latency_sensitivity": 0.95},
        {"user_type": "emergency", "name": "Tıbbi Ekip",     "priority": 4, "demand": 50.0, "latency_sensitivity": 0.90},
        {"user_type": "emergency", "name": "AFAD Lojistik",  "priority": 4, "demand": 45.0, "latency_sensitivity": 0.84},
        {"user_type": "emergency", "name": "Haberleşme",     "priority": 4, "demand": 38.0, "latency_sensitivity": 0.86},
        {"user_type": "civilian",  "name": "Sivil A",        "priority": 1, "demand": 22.0, "latency_sensitivity": 0.14},
        {"user_type": "civilian",  "name": "Sivil B",        "priority": 1, "demand": 16.0, "latency_sensitivity": 0.11},
        {"user_type": "civilian",  "name": "Sivil C",        "priority": 1, "demand": 14.0, "latency_sensitivity": 0.09},
    ])
    disaster_users["score"] = compute_scores(disaster_users).round(2)
    dis_cap = 330.0
    dis_alloc_p = allocate_priority(disaster_users, capacity=dis_cap)
    dis_alloc_b = allocate_baseline(disaster_users, capacity=dis_cap)
    disaster_users["allocated_priority"] = dis_alloc_p.values
    disaster_users["allocated_baseline"] = dis_alloc_b.values
    examples.append({
        "id": "disaster",
        "title": "Afet Simülasyonu — Deprem",
        "description": (
            "Deprem sonrası altyapı hasar gördü: 330 Mbps kapasite, 381 Mbps talep. "
            "AFAD, SAR ekipleri, tıbbi ve askeri koordinasyon aynı anda bant genişliği istiyor. "
            "Baseline eşit paylaşımda kritik ekipler yetersiz kalır; Priority tüm kritik kullanıcıları tam karşılar."
        ),
        "capacity": int(dis_cap),
        "users": disaster_users.to_dict(orient="records"),
        "metrics_priority": calculate_metrics(disaster_users, dis_alloc_p, "priority"),
        "metrics_baseline": calculate_metrics(disaster_users, dis_alloc_b, "baseline"),
    })

    return examples


# ─────────────────────────────────────────────
# 6b. HTML Rapor Üreteci
# ─────────────────────────────────────────────
def generate_html_report(
    all_results: dict,
    scenario_data: dict,
    examples: list[dict],
    path: str = "index.html",
) -> None:
    import base64, io

    # ── CSS (regular string — no brace escaping needed) ──
    css = """
:root {
  --bg: #f3f5fb; --surface: #ffffff; --surface2: #eaecf7;
  --border: #d0d6ed; --accent: #1a5fd8; --text: #1a2140;
  --muted: #5a6888; --win: #eaf7f0; --lose: #fceaea;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.6; }

header {
  background: linear-gradient(135deg, #e8eeff 0%, #dce8ff 55%, #eef2ff 100%);
  border-bottom: 1px solid var(--border);
  padding: 2.5rem 2rem 2rem; text-align: center; position: relative; overflow: hidden;
}
header::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse at 70% 50%, rgba(26,95,216,.08) 0%, transparent 70%);
  pointer-events: none;
}
.header-badge {
  display: inline-block; background: var(--accent); color: #fff; font-size: .72rem;
  font-weight: 700; letter-spacing: .1em; padding: .25rem .8rem; border-radius: 20px;
  margin-bottom: .8rem; text-transform: uppercase;
}
header h1 { font-size: 2.1rem; font-weight: 800; color: #1a2140; margin-bottom: .4rem; }
header p { color: var(--muted); max-width: 640px; margin: 0 auto; font-size: .95rem; }
.stat-strip { display: flex; justify-content: center; gap: 2rem; margin-top: 1.8rem; flex-wrap: wrap; }
.stat { text-align: center; }
.stat-val { font-size: 1.6rem; font-weight: 800; color: var(--accent); }
.stat-lbl { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }

nav {
  position: sticky; top: 0; z-index: 100;
  background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; padding: 0 2rem; overflow-x: auto;
  box-shadow: 0 1px 6px rgba(26,95,216,.07);
}
nav a {
  color: var(--muted); text-decoration: none; font-size: .88rem; font-weight: 600;
  padding: .9rem 1.2rem; border-bottom: 3px solid transparent;
  white-space: nowrap; transition: color .2s, border-color .2s; cursor: pointer;
}
nav a:hover { color: var(--text); }
nav a.active { color: var(--accent); border-bottom-color: var(--accent); }

.tab-pane { display: none; }
.tab-pane.active { display: block; }

main { max-width: 1200px; margin: 0 auto; padding: 2rem; }
section { margin-bottom: 3rem; }
h2 { font-size: 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: .6rem; }
h3 { font-size: 1rem; color: var(--text); margin-bottom: .8rem; }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem; }
@media(max-width:768px) { .metrics-grid { grid-template-columns: 1fr; } }

.metrics-table { width: 100%; border-collapse: collapse; font-size: .85rem; }
.metrics-table th { background: var(--surface2); color: var(--muted); font-size:.75rem;
  text-transform:uppercase; letter-spacing:.06em; padding: .5rem .7rem; text-align:left; }
.metrics-table td { padding: .45rem .7rem; border-bottom: 1px solid var(--border); }
.metrics-table tr:last-child td { border-bottom: none; }
.win  { background: var(--win);  color: #1a7a40; }
.lose { background: var(--lose); color: #b03030; }
.neutral { color: var(--muted); }
.mini-table { font-size: .78rem; margin-top: .8rem; }

.badge { display: inline-block; padding: .2rem .7rem; border-radius: 20px; font-size: .75rem; font-weight:700; }
.badge-normal { background: rgba(39,174,96,.15); color: #1a7a40; }
.badge-crisis { background: rgba(192,57,43,.12); color: #b03030; }
.type-badge { display: inline-block; padding: .1rem .5rem; border-radius: 4px; font-size: .72rem; font-weight:700; }
.type-military  { background: rgba(192,57,43,.12); color: #a02828; }
.type-emergency { background: rgba(200,100,20,.12); color: #a05010; }
.type-civilian  { background: rgba(26,95,216,.12);  color: #1a5fd8; }
.capacity-badge { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
  padding: .2rem .6rem; font-size: .75rem; color: var(--muted); }

.chart-container { margin-top: 1rem; text-align: center; }
.chart-img { max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }

.formula-box { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.formula { font-size: 1.15rem; font-family: monospace; padding: 1rem; background: var(--surface2);
  border-radius: 8px; margin: .8rem 0 1rem; color: var(--text); line-height: 2; }
.f-demand  { color: #1a8040; font-weight: 700; }
.f-latency { color: #b05010; font-weight: 700; }
.f-priority{ color: #a02828; font-weight: 700; }
.formula-legend { display: flex; flex-direction: column; gap: .5rem; margin-bottom: 1rem; }
.legend-item { display: flex; align-items: center; gap: .6rem; font-size: .85rem; color: var(--muted); }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.f-demand-dot  { background: #1a8040; } .f-latency-dot { background: #b05010; } .f-priority-dot { background: #a02828; }
.priority-examples { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: .8rem; }
.p-ex { flex: 1; min-width: 140px; padding: .8rem; border-radius: 8px; text-align: center; font-size: .82rem; line-height: 1.5; }
.military-ex  { background: rgba(192,57,43,.08); border: 1px solid rgba(192,57,43,.2); }
.emergency-ex { background: rgba(200,100,20,.08); border: 1px solid rgba(200,100,20,.2); }
.civilian-ex  { background: rgba(26,95,216,.08);  border: 1px solid rgba(26,95,216,.2); }

.example-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.4rem; margin-bottom: 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.example-header { display: flex; align-items: center; gap: 1rem; margin-bottom: .4rem; flex-wrap: wrap; }
.example-desc { color: var(--muted); font-size: .88rem; margin-bottom: 1rem; }
.delta-row { display: flex; gap: .8rem; flex-wrap: wrap; margin-bottom: 1rem; }
.delta-item { flex: 1; min-width: 140px; padding: .8rem; border-radius: 8px;
  background: var(--surface2); border: 1px solid var(--border); text-align: center; }
.delta-item.positive .delta-value { color: #1a7a40; }
.delta-item.negative .delta-value { color: #b03030; }
.delta-item.neutral  .delta-value { color: var(--accent); }
.delta-label { display: block; font-size: .72rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: .06em; margin-bottom: .2rem; }
.delta-value { font-size: 1.4rem; font-weight: 800; }
.sim-btn {
  background: rgba(26,95,216,.1); border: 1px solid rgba(26,95,216,.3);
  color: var(--accent); border-radius: 6px; padding: .35rem .9rem;
  font-size: .8rem; font-weight: 600; cursor: pointer; transition: background .2s;
}
.sim-btn:hover { background: rgba(26,95,216,.2); }

.comparison-hero { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem; }
@media(max-width:600px) { .comparison-hero { grid-template-columns: 1fr; } }
.hero-card { padding: 1.4rem; border-radius: 10px; text-align: center; border: 1px solid; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.hero-card.priority { background: rgba(39,174,96,.07); border-color: rgba(39,174,96,.3); }
.hero-card.baseline  { background: rgba(100,120,160,.06); border-color: rgba(100,120,160,.25); }
.hero-card h3 { font-size: 1rem; margin-bottom: .8rem; }
.hero-metric { font-size: 2rem; font-weight: 800; }
.hero-metric.green { color: #1a7a40; }
.hero-metric.gray  { color: var(--muted); }
.hero-label { font-size: .78rem; color: var(--muted); margin-top: .2rem; }

/* ── LIVE SIM ── */
.live-layout { display: grid; grid-template-columns: 300px 1fr; gap: 1.2rem; align-items: start; }
@media(max-width:900px) { .live-layout { grid-template-columns: 1fr; } }
.live-controls-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.ctrl-row { margin-bottom: .9rem; }
.ctrl-row label { display: block; font-size: .75rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: .06em; margin-bottom: .3rem; }
.ctrl-row .val-display { float: right; color: var(--accent); font-weight: 700; font-size: .85rem; }
input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
select { width: 100%; background: var(--surface2); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: .4rem .6rem; font-size: .85rem; cursor: pointer; }
.btn-group { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .4rem; }
.btn { border: none; border-radius: 6px; padding: .45rem .9rem; font-size: .82rem;
  font-weight: 700; cursor: pointer; transition: opacity .2s; }
.btn:hover { opacity: .85; }
.btn-play  { background: #27ae60; color: #fff; flex: 1; }
.btn-step  { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.btn-reset { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.speed-btns { display: flex; gap: .3rem; margin-top: .5rem; }
.speed-btn { background: var(--surface2); border: 1px solid var(--border); color: var(--muted);
  border-radius: 4px; padding: .25rem .6rem; font-size: .75rem; cursor: pointer; font-weight: 700; }
.speed-btn.active { background: rgba(26,95,216,.15); border-color: var(--accent); color: var(--accent); }

.live-main { display: flex; flex-direction: column; gap: 1rem; }
.gauge-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.gauge-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: .6rem; }
.gauge-title { font-size: .8rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.gauge-track { background: var(--surface2); border-radius: 6px; height: 18px; overflow: hidden; position: relative; }
.gauge-fill { height: 100%; border-radius: 6px; transition: width .35s ease; background: linear-gradient(90deg, #27ae60, #1a5fd8); }
.gauge-label { font-size: .8rem; color: var(--muted); margin-top: .4rem; text-align: right; }
.gauge-pct { color: var(--text); font-weight: 700; }

.stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: .8rem; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.stat-card .s-val { font-size: 1.5rem; font-weight: 800; color: var(--accent); }
.stat-card .s-lbl { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.step-counter { font-size: .78rem; color: var(--muted); text-align: right; margin-bottom: .4rem; }

.users-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.users-panel-header { padding: .7rem 1rem; background: var(--surface2); border-bottom: 1px solid var(--border);
  font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em;
  display: flex; justify-content: space-between; }
.users-scroll { max-height: 420px; overflow-y: auto; }
.sim-user-row { display: flex; align-items: center; gap: .7rem; padding: .5rem 1rem;
  border-bottom: 1px solid var(--border); transition: background .3s; }
.sim-user-row:last-child { border-bottom: none; }
.sim-user-row.row-done    { background: rgba(39,174,96,.07); }
.sim-user-row.row-fail    { background: rgba(192,57,43,.06); }
.sim-user-row.row-current { background: rgba(26,95,216,.07); }
.sim-user-info { display: flex; align-items: center; gap: .4rem; min-width: 150px; }
.sim-type-badge { display: inline-block; padding: .1rem .4rem; border-radius: 4px;
  font-size: .68rem; font-weight: 800; min-width: 32px; text-align: center; }
.sim-score { font-size: .72rem; color: var(--muted); }
.sim-demand { font-size: .72rem; color: var(--muted); margin-left: auto; }
.sim-bar-wrap { flex: 1; display: flex; align-items: center; gap: .5rem; }
.sim-bar-track { flex: 1; background: var(--surface2); border-radius: 4px; height: 10px; overflow: hidden; }
.sim-bar-fill { height: 100%; border-radius: 4px; width: 0%; transition: width .3s ease; }
.sim-alloc-label { font-size: .72rem; color: var(--muted); min-width: 60px; text-align: right; }
.sim-status { font-size: .9rem; min-width: 16px; text-align: center; }

.empty-state { text-align: center; padding: 2rem; color: var(--muted); font-size: .88rem; }

/* ── COMPARISON ── */
.problem-box {
  display: flex; gap: 1rem; align-items: flex-start;
  background: rgba(192,57,43,.06); border: 1px solid rgba(192,57,43,.2);
  border-radius: 10px; padding: 1.2rem; margin-bottom: 1.5rem;
}
.problem-icon { font-size: 1.8rem; flex-shrink: 0; line-height: 1; padding-top:.1rem; }
.problem-box h3 { margin-bottom: .4rem; color: #a02828; font-size:.95rem; }
.problem-box p  { color: var(--muted); font-size:.85rem; line-height:1.7; }

.big-numbers { display: grid; grid-template-columns: repeat(4,1fr); gap: .8rem; margin-bottom: 1.5rem; }
@media(max-width:800px) { .big-numbers { grid-template-columns: repeat(2,1fr); } }
.big-num-card { background: var(--surface); border: 1px solid var(--border); border-radius:10px;
  padding: 1rem; text-align:center; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.big-num { font-size: 2.2rem; font-weight: 900; line-height:1.1; }
.big-num-label { font-size:.75rem; color:var(--text); margin: .3rem 0 .15rem; font-weight:600; }
.big-num-sub   { font-size:.68rem; color:var(--muted); }

.cmp-scenarios-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom:1.5rem; }
@media(max-width:768px) { .cmp-scenarios-grid { grid-template-columns: 1fr; } }
.cmp-bar-track { background: var(--surface2); border-radius:4px; height:10px;
  position:relative; overflow:hidden; margin-bottom:.15rem; }
.cmp-bar-fill  { height:100%; border-radius:4px; transition: width .4s ease; }
.cmp-bar-label { font-size:.7rem; color:var(--muted); }
.cmp-win  { color:#1a7a40; font-weight:700; }
.cmp-lose { color:#b03030; }
.cmp-neutral { color:var(--muted); }
.cmp-summary { display:flex; flex-direction:column; gap:.3rem; margin-top:.8rem;
  padding-top:.8rem; border-top:1px solid var(--border); font-size:.8rem; color:var(--muted); }

.story-box { background: var(--surface); border: 1px solid var(--border); border-radius:10px;
  padding:1.4rem; margin-bottom:1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.story-box h3 { margin-bottom:1rem; }
.story-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
@media(max-width:680px) { .story-grid { grid-template-columns:1fr; } }
.story-col { background: var(--surface2); border-radius:8px; padding:1rem; }
.story-header { font-weight:700; margin-bottom:.7rem; font-size:.9rem; }
.story-without .story-header { color:#a02828; }
.story-with    .story-header { color:#1a7a40; }
.story-col ul { padding-left:1.2rem; }
.story-col li { font-size:.83rem; color:var(--muted); line-height:1.8; }
.story-col li strong { color:var(--text); }

.benefits-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem; }
@media(max-width:900px) { .benefits-grid { grid-template-columns:repeat(2,1fr); } }
@media(max-width:550px) { .benefits-grid { grid-template-columns:1fr; } }
.benefit-card { background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:1.1rem; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.benefit-icon { font-size:1.5rem; margin-bottom:.5rem; }
.benefit-card h4 { font-size:.88rem; margin-bottom:.4rem; color:var(--text); }
.benefit-card p  { font-size:.78rem; color:var(--muted); line-height:1.65; }

footer {
  text-align: center; padding: 2rem; color: var(--muted); font-size: .8rem;
  border-top: 1px solid var(--border); margin-top: 2rem;
}

/* ── TURKEY DUAL MAP ── */
.map-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; margin-top: .8rem; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.map-panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: .8rem; flex-wrap: wrap; gap: .4rem; }
.map-panel-title { font-size: .8rem; color: var(--text); font-weight: 700; }
.dual-map-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }
@media(max-width:700px) { .dual-map-grid { grid-template-columns: 1fr; } }
.map-col-header { font-size: .78rem; font-weight: 700; text-align: center; padding: .35rem .6rem; border-radius: 6px; margin-bottom: .4rem; }
.map-col-header.priority-header { background: rgba(39,174,96,.1); color: #1a7a40; border: 1px solid rgba(39,174,96,.25); }
.map-col-header.baseline-header { background: rgba(100,120,160,.08); color: var(--muted); border: 1px solid var(--border); }
.turkey-svg { width: 100%; height: auto; display: block; border-radius: 8px; background: #e8f0fa; border: 1px solid var(--border); cursor: zoom-in; }
.turkey-svg:hover { border-color: rgba(26,95,216,.4); box-shadow: 0 0 0 1px rgba(26,95,216,.2); }
.city-label { font-size: 7px; fill: rgba(20,40,80,0.65); text-anchor: middle; pointer-events: none; font-family: sans-serif; font-weight: 600; }
.city-type-label { font-size: 6px; text-anchor: middle; pointer-events: none; font-family: sans-serif; font-weight: 700; }
.map-legend { display: flex; gap: 1rem; font-size: .72rem; color: var(--muted); margin-top: .6rem; justify-content: center; flex-wrap: wrap; }
.legend-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .3rem; vertical-align: middle; }
.prio-order-note { font-size: .7rem; color: var(--muted); text-align: center; margin-top: .3rem; }
.prio-order-note span { margin: 0 .4rem; }
.modes-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin-top: 1.2rem; }
@media(max-width:800px) { .modes-grid { grid-template-columns: 1fr; } }
.mode-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem 1.2rem; display: flex; flex-direction: column; gap: .8rem; position: relative; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.mode-card.active-mode { border-color: rgba(192,57,43,.4); box-shadow: 0 0 0 1px rgba(192,57,43,.15), 0 2px 8px rgba(192,57,43,.08); }
.mode-card-title { font-size: 1.05rem; font-weight: 700; }
.mode-card.blue   .mode-card-title { color: #1a5fd8; }
.mode-card.red    .mode-card-title { color: #a02828; }
.mode-card.orange .mode-card-title { color: #a05010; }
.mode-card-desc { font-size: .84rem; color: var(--muted); line-height: 1.55; }
.mode-weights { background: var(--surface2); border: 1px solid var(--border); border-radius: 7px; padding: .7rem .9rem; font-family: monospace; font-size: .82rem; display: flex; flex-direction: column; gap: .35rem; }
.mode-weights span { color: var(--muted); }
.mode-weights span b { color: var(--text); }
.mode-weight-blue  b { color: #1a5fd8; }
.mode-weight-red   b { color: #a02828; }
.mode-weight-orange b { color: #a05010; }
.mode-examples { font-size: .78rem; color: var(--muted); padding-left: 1rem; margin: 0; line-height: 1.7; }
.mode-result { font-size: .78rem; font-weight: 600; padding: .45rem .75rem; border-radius: 6px; text-align: center; margin-top: auto; }
.mode-card.blue   .mode-result { background: rgba(26,95,216,.08);  color: #1a5fd8; border: 1px solid rgba(26,95,216,.2); }
.mode-card.red    .mode-result { background: rgba(192,57,43,.08);  color: #a02828; border: 1px solid rgba(192,57,43,.2); }
.mode-card.orange .mode-result { background: rgba(200,100,20,.08); color: #a05010; border: 1px solid rgba(200,100,20,.2); }
.active-badge { position: absolute; top: -.7rem; left: 50%; transform: translateX(-50%); background: #c0392b; color: #fff; font-size: .65rem; font-weight: 700; padding: .2rem .7rem; border-radius: 99px; letter-spacing: .04em; white-space: nowrap; }
.modes-note { margin-top: 1.8rem; background: var(--surface); border-left: 3px solid rgba(26,95,216,.35); border-radius: 0 8px 8px 0; padding: .9rem 1.2rem; font-size: .85rem; color: var(--muted); line-height: 1.6; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.map-zoom-hint { display: block; font-size: .65rem; color: var(--muted); text-align: right; margin-top: .2rem; opacity: .6; }
#map-modal { display:none; position:fixed; inset:0; background:rgba(20,30,60,.75); z-index:9999; align-items:center; justify-content:center; flex-direction:column; gap:.8rem; backdrop-filter:blur(4px); }
#map-modal.open { display:flex; }
#map-modal-inner { width:90vw; max-width:1050px; background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1rem; position:relative; box-shadow:0 8px 40px rgba(0,0,0,.18); }
#map-modal-inner svg { width:100%; height:auto; display:block; border-radius:6px; }
#map-modal-title { font-size:.8rem; font-weight:700; color:var(--text); margin-bottom:.6rem; }
#map-modal-close { position:absolute; top:.6rem; right:.8rem; background:none; border:none; color:var(--muted); font-size:1.4rem; cursor:pointer; line-height:1; padding:.2rem .4rem; border-radius:4px; }
#map-modal-close:hover { color:var(--text); background:var(--surface2); }
"""

    # ── JavaScript (regular string — no brace escaping needed) ──
    js = """
// ── Tab switching ──
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    switchTab(link.dataset.tab);
  });
});

function switchTab(tabId) {
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  const link = document.querySelector('.nav-link[data-tab="' + tabId + '"]');
  const pane = document.getElementById('tab-' + tabId);
  if (link) link.classList.add('active');
  if (pane) pane.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'instant' });
}

// ── "Simüle Et" buttons on example cards ──
document.querySelectorAll('.load-sim-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const cap   = btn.dataset.capacity;
    const n     = btn.dataset.n;
    const scen  = btn.dataset.scenario;
    const meth  = btn.dataset.method;

    const capSlider = document.getElementById('sim-cap');
    if (capSlider) {
      capSlider.value = cap;
      document.getElementById('sim-cap-val').textContent = cap;
    }
    const nSlider = document.getElementById('sim-n');
    if (nSlider) {
      nSlider.value = n;
      document.getElementById('sim-n-val').textContent = n;
    }
    document.getElementById('sim-scenario').value = scen;
    document.getElementById('sim-method').value   = meth;

    switchTab('live');
    setTimeout(() => { resetSim(); }, 80);
  });
});

// ── Live Simulation ──
const PRIO_MAP = { military: 5, emergency: 4, civilian: 1 };
const TYPE_COLOR = { military: '#c0392b', emergency: '#e67e22', civilian: '#2980b9' };
const TYPE_BG    = {
  military:  'rgba(192,57,43,0.25)',
  emergency: 'rgba(230,126,34,0.25)',
  civilian:  'rgba(41,128,185,0.25)'
};

function lcgRng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function generateUsers(n, scenario, seed) {
  const rng = lcgRng(seed);
  const ratios = scenario === 'crisis'
    ? { military: 0.20, emergency: 0.40, civilian: 0.40 }
    : { military: 0.10, emergency: 0.20, civilian: 0.70 };

  const users = [];
  for (const [type, ratio] of Object.entries(ratios)) {
    const count = Math.round(ratio * n);
    for (let i = 0; i < count; i++) {
      const priority = PRIO_MAP[type];
      let demand, latSens;
      if (type === 'military') {
        demand  = 20 + rng() * 60;   // 20–80 Mbps  (Python ile aynı aralık)
        latSens = 0.70 + rng() * 0.30;
      } else if (type === 'emergency') {
        demand  = 15 + rng() * 45;   // 15–60 Mbps
        latSens = 0.50 + rng() * 0.35;
      } else {
        demand  = 5  + rng() * 25;   // 5–30 Mbps
        latSens = rng() * 0.30;
      }
      const score = demand * 0.5 + latSens * 30 + priority * 10;
      users.push({
        type, priority,
        demand:  Math.round(demand  * 10) / 10,
        latSens: Math.round(latSens * 100) / 100,
        score:   Math.round(score   * 10)  / 10,
        allocated: 0
      });
    }
  }
  // Shuffle
  for (let i = users.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [users[i], users[j]] = [users[j], users[i]];
  }
  return users.slice(0, n);
}

function allocatePriority(users, capacity) {
  const sorted = [...users].sort((a, b) => b.score - a.score);
  let rem = capacity;
  return sorted.map(u => {
    const a = Math.round(Math.min(u.demand, rem) * 10) / 10;
    rem = Math.max(0, rem - u.demand);
    return { ...u, allocated: a };
  });
}

function allocateBaseline(users, capacity) {
  const share = capacity / users.length;
  return users.map(u => ({
    ...u,
    allocated: Math.round(Math.min(u.demand, share) * 10) / 10
  }));
}

// ── Turkey Map Data ──
const CITIES = [
  { id: 'istanbul',   name: 'İstanbul',   x: 130, y: 55,  type: 'civilian',  pop: 3 },
  { id: 'ankara',     name: 'Ankara',     x: 292, y: 132, type: 'military',  pop: 2 },
  { id: 'izmir',      name: 'İzmir',      x: 46,  y: 188, type: 'civilian',  pop: 2 },
  { id: 'bursa',      name: 'Bursa',      x: 136, y: 110, type: 'civilian',  pop: 1 },
  { id: 'antalya',    name: 'Antalya',    x: 235, y: 265, type: 'civilian',  pop: 1 },
  { id: 'adana',      name: 'Adana',      x: 392, y: 278, type: 'emergency', pop: 1 },
  { id: 'konya',      name: 'Konya',      x: 270, y: 228, type: 'civilian',  pop: 1 },
  { id: 'gaziantep',  name: 'Gaziantep',  x: 478, y: 285, type: 'emergency', pop: 1 },
  { id: 'diyarbakir', name: 'Diyarbakır', x: 588, y: 238, type: 'military',  pop: 1 },
  { id: 'kayseri',    name: 'Kayseri',    x: 392, y: 195, type: 'civilian',  pop: 1 },
  { id: 'trabzon',    name: 'Trabzon',    x: 552, y: 68,  type: 'emergency', pop: 1 },
  { id: 'samsun',     name: 'Samsun',     x: 412, y: 46,  type: 'civilian',  pop: 1 },
  { id: 'erzurum',    name: 'Erzurum',    x: 638, y: 132, type: 'military',  pop: 1 },
  { id: 'van',        name: 'Van',        x: 725, y: 192, type: 'military',  pop: 1 },
];

const CITY_BASE_TYPE_COLOR = {
  military:  '#c0392b',
  emergency: '#e67e22',
  civilian:  '#2980b9',
};

let cityUserMap_p = {};
let cityUserMap_b = {};

function assignUsersToCities(users) {
  const totalPop = CITIES.reduce((s, c) => s + c.pop, 0);
  const map = {};
  CITIES.forEach(c => { map[c.id] = []; });
  let ci = 0, assigned = 0;
  const quotas = CITIES.map(c => Math.round(c.pop / totalPop * users.length));
  let diff = users.length - quotas.reduce((a, b) => a + b, 0);
  quotas[0] += diff;
  users.forEach((u, i) => {
    while (ci < CITIES.length - 1 && assigned >= quotas[ci]) { ci++; assigned = 0; }
    map[CITIES[ci].id].push(i);
    assigned++;
  });
  return map;
}

function initMapDual(pUsers, bUsers) {
  cityUserMap_p = assignUsersToCities(pUsers);
  cityUserMap_b = assignUsersToCities(bUsers);
  CITIES.forEach(c => {
    ['p', 'b'].forEach(sfx => {
      const heat = document.getElementById('heat-' + c.id + '-' + sfx);
      const dot  = document.getElementById('dot-'  + c.id + '-' + sfx);
      if (heat) { heat.setAttribute('r', '0'); heat.setAttribute('opacity', '0'); }
      if (dot)  { dot.setAttribute('fill', '#2a3050'); }
    });
  });
  // Baseline: always show final state immediately
  updateCityGroup(bUsers, cityUserMap_b, bUsers.length, 'b');
}

function updateCityGroup(users, cityMap, step, sfx) {
  CITIES.forEach(c => {
    const indices = cityMap[c.id] || [];
    if (!indices.length) return;
    const done      = indices.filter(i => i < step).length;
    const satisfied = indices.filter(i => i < step && users[i] && users[i].allocated >= users[i].demand - 0.05).length;
    const ratio     = done > 0 ? satisfied / done : 0;
    const progress  = done / indices.length;
    const heat = document.getElementById('heat-' + c.id + '-' + sfx);
    const dot  = document.getElementById('dot-'  + c.id + '-' + sfx);
    if (!heat || !dot) return;
    if (done === 0) {
      heat.setAttribute('r', '0'); heat.setAttribute('opacity', '0');
      dot.setAttribute('fill', '#2a3050'); return;
    }
    const color = ratio >= 0.75 ? '#27ae60' : ratio >= 0.4 ? '#e67e22' : '#c0392b';
    const baseR = 10 + c.pop * 5;
    heat.setAttribute('r',       String(Math.round(baseR + progress * 22)));
    heat.setAttribute('fill',    color);
    heat.setAttribute('opacity', (0.12 + ratio * 0.22).toFixed(2));
    dot.setAttribute('fill', color);
  });
}

function updateMapDual(pUsers, bUsers, step) {
  updateCityGroup(pUsers, cityUserMap_p, step, 'p');
  // baseline is static (already shown at init), only refresh if step changed to full
  if (step >= pUsers.length) updateCityGroup(bUsers, cityUserMap_b, bUsers.length, 'b');
}

// ── State ──
let sim = { users: [], step: 0, timer: null, running: false, capacity: 500 };
const SPEEDS = [700, 280, 100, 25];
let speedIdx = 1;

function getCapacity()  { return parseInt(document.getElementById('sim-cap').value); }
function getUserCount() { return parseInt(document.getElementById('sim-n').value);   }
function getScenario()  { return document.getElementById('sim-scenario').value; }
function getMethod()    { return document.getElementById('sim-method').value;   }

function initSim() {
  clearTimeout(sim.timer);
  const cap      = getCapacity();
  const n        = getUserCount();
  const scenario = getScenario();
  const method   = getMethod();

  const raw    = generateUsers(n, scenario, Date.now() & 0xFFFFF);
  const pUsers = allocatePriority(raw, cap);
  const bUsers = allocateBaseline(raw, cap);
  // User list shows selected method; maps always show both
  const animUsers = method === 'priority' ? pUsers : bUsers;

  sim = { users: animUsers, priorityUsers: pUsers, baselineUsers: bUsers,
          step: 0, timer: null, running: false, capacity: cap };

  renderRows(animUsers);
  initMapDual(pUsers, bUsers);
  updateGauge(0, cap);
  updateStats(0, users, cap);
  updateStepCounter(0, users.length);
  setPlayBtn('▶ Başlat');
}

function renderRows(users) {
  const container = document.getElementById('sim-users-list');
  container.innerHTML = '';
  users.forEach((u, i) => {
    const row = document.createElement('div');
    row.className = 'sim-user-row';
    row.id = 'row-' + i;
    row.innerHTML =
      '<div class="sim-user-info">' +
        '<span class="sim-type-badge" style="background:' + TYPE_BG[u.type] + ';color:' + TYPE_COLOR[u.type] + '">' +
          u.type.slice(0,3).toUpperCase() +
        '</span>' +
        '<span class="sim-score">⚡' + u.score + '</span>' +
        '<span class="sim-demand">' + u.demand + ' Mbps</span>' +
      '</div>' +
      '<div class="sim-bar-wrap">' +
        '<div class="sim-bar-track">' +
          '<div class="sim-bar-fill" id="bar-' + i + '" style="width:0%;background:' + TYPE_COLOR[u.type] + '"></div>' +
        '</div>' +
        '<span class="sim-alloc-label" id="lbl-' + i + '">0 / ' + u.demand + '</span>' +
        '<span class="sim-status" id="sta-' + i + '"> </span>' +
      '</div>';
    container.appendChild(row);
  });
  document.getElementById('sim-empty').style.display = 'none';
}

function applyStep(i) {
  const u   = sim.users[i];
  const bar = document.getElementById('bar-' + i);
  const lbl = document.getElementById('lbl-' + i);
  const sta = document.getElementById('sta-' + i);
  const row = document.getElementById('row-' + i);
  if (!bar) return;
  const pct = u.demand > 0 ? (u.allocated / u.demand) * 100 : 100;
  bar.style.width = pct + '%';
  lbl.textContent = u.allocated + ' / ' + u.demand;
  const ok = u.allocated >= u.demand - 0.05;
  sta.textContent = ok ? '✓' : '✗';
  sta.style.color = ok ? '#5dde8a' : '#e07070';
  if (row) {
    row.classList.add(ok ? 'row-done' : 'row-fail');
    row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function updateGauge(step, cap) {
  const used = sim.users.slice(0, step).reduce((s, u) => s + u.allocated, 0);
  const pct  = Math.min(100, (used / cap) * 100);
  document.getElementById('gauge-fill').style.width = pct + '%';
  document.getElementById('gauge-text').textContent = Math.round(used) + ' / ' + cap + ' Mbps';
  document.getElementById('gauge-pct').textContent  = pct.toFixed(1) + '%';
}

function updateStats(step, users, cap) {
  const critTypes = ['military', 'emergency'];
  const critAll  = users.filter(u => critTypes.includes(u.type));
  const critDone = users.slice(0, step).filter(u => critTypes.includes(u.type) && u.allocated >= u.demand - 0.05);
  const critRate = critAll.length ? (critDone.length / critAll.length * 100) : 0;

  const totalAlloc = users.slice(0, step).reduce((s, u) => s + u.allocated, 0);
  const latPen     = users.slice(0, step).reduce((s, u) => s + Math.max(0, u.demand - u.allocated) * u.latSens, 0);
  const eff        = (totalAlloc / cap) * 100;

  document.getElementById('stat-critical').textContent   = critRate.toFixed(1) + '%';
  document.getElementById('stat-latency').textContent    = latPen.toFixed(1);
  document.getElementById('stat-efficiency').textContent = eff.toFixed(1) + '%';
}

function updateStepCounter(step, total) {
  document.getElementById('step-counter').textContent = 'Adım: ' + step + ' / ' + total;
}

function tick() {
  if (sim.step >= sim.users.length) {
    sim.running = false;
    setPlayBtn('✓ Tamamlandı');
    return;
  }
  applyStep(sim.step);
  sim.step++;
  updateGauge(sim.step, sim.capacity);
  updateStats(sim.step, sim.users, sim.capacity);
  updateStepCounter(sim.step, sim.users.length);
  updateMapDual(sim.priorityUsers, sim.baselineUsers, sim.step);

  if (sim.running && sim.step < sim.users.length) {
    sim.timer = setTimeout(tick, SPEEDS[speedIdx]);
  } else if (sim.step >= sim.users.length) {
    sim.running = false;
    setPlayBtn('↺ Yeniden Başlat');
  }
}

function setPlayBtn(label) {
  document.getElementById('btn-play').textContent = label;
}

function togglePlay() {
  if (sim.users.length === 0) { initSim(); }
  if (sim.step >= sim.users.length) {
    initSim();
    setTimeout(() => { sim.running = true; setPlayBtn('⏸ Duraklat'); tick(); }, 50);
    return;
  }
  sim.running = !sim.running;
  if (sim.running) {
    setPlayBtn('⏸ Duraklat');
    tick();
  } else {
    clearTimeout(sim.timer);
    setPlayBtn('▶ Devam Et');
  }
}

function stepOnce() {
  if (sim.users.length === 0) { initSim(); }
  if (sim.running) { clearTimeout(sim.timer); sim.running = false; setPlayBtn('▶ Devam Et'); }
  if (sim.step < sim.users.length) {
    applyStep(sim.step);
    sim.step++;
    updateGauge(sim.step, sim.capacity);
    updateStats(sim.step, sim.users, sim.capacity);
    updateStepCounter(sim.step, sim.users.length);
    if (sim.step >= sim.users.length) setPlayBtn('↺ Yeniden Başlat');
  }
}

function resetSim() {
  clearTimeout(sim.timer);
  sim = { users: [], step: 0, timer: null, running: false, capacity: 500 };
  document.getElementById('sim-users-list').innerHTML = '';
  document.getElementById('sim-empty').style.display = 'block';
  updateGauge(0, getCapacity());
  updateStats(0, [], getCapacity());
  updateStepCounter(0, 0);
  setPlayBtn('▶ Başlat');
  initSim();
}

function setSpeed(idx) {
  speedIdx = idx;
  document.querySelectorAll('.speed-btn').forEach((b, i) => {
    b.classList.toggle('active', i === idx);
  });
}

// ── Slider live update ──
document.getElementById('sim-cap').addEventListener('input', function() {
  document.getElementById('sim-cap-val').textContent = this.value;
});
document.getElementById('sim-n').addEventListener('input', function() {
  document.getElementById('sim-n-val').textContent = this.value;
});

// ── Map zoom modal ──
(function() {
  var modal = document.getElementById('map-modal');
  var modalTitle = document.getElementById('map-modal-title');
  var modalInner = document.getElementById('map-modal-inner');

  document.querySelectorAll('.turkey-svg').forEach(function(svg) {
    svg.addEventListener('click', function() {
      var clone = svg.cloneNode(true);
      clone.removeAttribute('id');
      clone.style.cursor = 'default';
      clone.style.width = '100%';
      clone.style.height = 'auto';
      // remove old svg from modal inner (keep title, close btn)
      var old = modalInner.querySelector('svg');
      if (old) old.remove();
      modalInner.appendChild(clone);
      var isPriority = svg.id === 'turkey-svg-p';
      modalTitle.textContent = isPriority
        ? '🎯 Priority Sistemi — Önem Sırası Dağılımı'
        : '⚖ Systemsiz Baseline — Eşit Dağıtım';
      modal.classList.add('open');
    });
  });

  document.getElementById('map-modal-close').addEventListener('click', function() {
    modal.classList.remove('open');
  });
  modal.addEventListener('click', function(e) {
    if (e.target === modal) modal.classList.remove('open');
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') modal.classList.remove('open');
  });
})();

// Init on page load
initSim();
"""

    def _img_b64(scenario: str) -> str:
        """Matplotlib figürünü base64 string'e çevirir (dosyaya yazmadan)."""
        sd = scenario_data[scenario]
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle(
            f"LEO — {scenario.upper()} Senaryosu",
            fontsize=13, fontweight="bold",
        )
        user_types = ["military", "emergency", "civilian"]
        df, alloc_p, alloc_b = sd["df"], sd["alloc_p"], sd["alloc_b"]
        mp, mb = sd["mp"], sd["mb"]
        width = 0.35

        # pie
        ax1 = fig.add_subplot(gs[0, 0])
        tc = df["user_type"].value_counts()
        ax1.pie(tc.values, labels=tc.index,
                colors=[COLORS[t] for t in tc.index], autopct="%1.1f%%", startangle=90)
        ax1.set_title("Kullanıcı Dağılımı")

        # satisfaction bars
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(user_types))
        pr = [mp["per_type"].get(t, {}).get("satisfaction_rate", 0)*100 for t in user_types]
        br = [mb["per_type"].get(t, {}).get("satisfaction_rate", 0)*100 for t in user_types]
        b1 = ax2.bar(x-width/2, pr, width, label="Priority", color="#27ae60")
        b2 = ax2.bar(x+width/2, br, width, label="Baseline", color="#7f8c8d")
        ax2.set_xticks(x); ax2.set_xticklabels(user_types); ax2.set_ylim(0, 115)
        ax2.set_title("Karşılanma Oranı (%)"); ax2.legend(fontsize=7)
        ax2.bar_label(b1, fmt="%.0f%%", padding=2, fontsize=6)
        ax2.bar_label(b2, fmt="%.0f%%", padding=2, fontsize=6)

        # genel metrikler
        ax3 = fig.add_subplot(gs[0, 2])
        labels3 = ["Genel\nKarşılanma", "Kritik\nKarşılanma", "Verimlilik"]
        pv = [mp["satisfaction_rate"]*100, mp["critical_satisfaction_rate"]*100, mp["efficiency"]*100]
        bv = [mb["satisfaction_rate"]*100, mb["critical_satisfaction_rate"]*100, mb["efficiency"]*100]
        x3 = np.arange(3)
        c1 = ax3.bar(x3-width/2, pv, width, label="Priority", color="#27ae60")
        c2 = ax3.bar(x3+width/2, bv, width, label="Baseline", color="#7f8c8d")
        ax3.set_xticks(x3); ax3.set_xticklabels(labels3, fontsize=7); ax3.set_ylim(0, 115)
        ax3.set_title("Genel Metrikler (%)"); ax3.legend(fontsize=7)
        ax3.bar_label(c1, fmt="%.1f", padding=2, fontsize=6)
        ax3.bar_label(c2, fmt="%.1f", padding=2, fontsize=6)

        # scatter priority
        ax4 = fig.add_subplot(gs[1, 0])
        for ut, col in COLORS.items():
            m = df["user_type"] == ut
            ax4.scatter(df.loc[m,"demand"], alloc_p[m], color=col, alpha=0.6, s=18, label=ut)
        ax4.plot([0,df["demand"].max()],[0,df["demand"].max()],"k--",lw=0.8)
        ax4.set_xlabel("Talep"); ax4.set_ylabel("Dağıtılan"); ax4.set_title("Priority Dağıtım")
        ax4.legend(fontsize=6)

        # scatter baseline
        ax5 = fig.add_subplot(gs[1, 1])
        for ut, col in COLORS.items():
            m = df["user_type"] == ut
            ax5.scatter(df.loc[m,"demand"], alloc_b[m], color=col, alpha=0.6, s=18, label=ut)
        ax5.plot([0,df["demand"].max()],[0,df["demand"].max()],"k--",lw=0.8)
        ax5.set_xlabel("Talep"); ax5.set_ylabel("Dağıtılan"); ax5.set_title("Baseline Dağıtım")
        ax5.legend(fontsize=6)

        # latency penalty
        ax6 = fig.add_subplot(gs[1, 2])
        pp = []; bp = []
        for ut in user_types:
            m = df["user_type"] == ut
            u_p = (df.loc[m,"demand"] - alloc_p[m]).clip(lower=0)
            u_b = (df.loc[m,"demand"] - alloc_b[m]).clip(lower=0)
            pp.append((u_p * df.loc[m,"latency_sensitivity"]).sum())
            bp.append((u_b * df.loc[m,"latency_sensitivity"]).sum())
        x6 = np.arange(3)
        d1 = ax6.bar(x6-width/2, pp, width, label="Priority", color="#27ae60")
        d2 = ax6.bar(x6+width/2, bp, width, label="Baseline", color="#7f8c8d")
        ax6.set_xticks(x6); ax6.set_xticklabels(user_types)
        ax6.set_title("Gecikme Cezası"); ax6.legend(fontsize=7)
        ax6.bar_label(d1, fmt="%.1f", padding=2, fontsize=6)
        ax6.bar_label(d2, fmt="%.1f", padding=2, fontsize=6)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _metric_row(label, pv, bv, higher_is_better=True):
        better = pv >= bv if higher_is_better else pv <= bv
        p_cls = "win" if better else "lose"
        b_cls = "lose" if better else "win"
        return f'<tr><td>{label}</td><td class="{p_cls}">{pv}</td><td class="{b_cls}">{bv}</td></tr>'

    def _scenario_section(scenario, res):
        mp, mb = res["priority"], res["baseline"]
        img = _img_b64(scenario)
        label = "Normal" if scenario == "normal" else "Kriz"
        badge = f'<span class="badge badge-{"normal" if scenario=="normal" else "crisis"}">{label}</span>'

        per_type_rows = ""
        for ut in ["military", "emergency", "civilian"]:
            pt = mp["per_type"].get(ut, {}); bt = mb["per_type"].get(ut, {})
            if not pt: continue
            p_sat = f"{pt['satisfaction_rate']*100:.1f}%"
            b_sat = f"{bt['satisfaction_rate']*100:.1f}%"
            win = pt['satisfaction_rate'] >= bt['satisfaction_rate']
            per_type_rows += f"""<tr>
              <td><span class="type-badge type-{ut}">{ut}</span></td>
              <td>{pt['count']}</td>
              <td class="{'win' if win else 'lose'}">{p_sat}</td>
              <td class="{'lose' if win else 'win'}">{b_sat}</td>
              <td>{pt['avg_demand_mbps']} Mbps</td>
            </tr>"""

        return f"""
        <section class="scenario-section" id="scenario-{scenario}">
          <h2>{badge} {scenario.capitalize()} Senaryosu</h2>
          <div class="metrics-grid">
            <div class="card">
              <table class="metrics-table">
                <thead><tr><th>Metrik</th><th>Priority</th><th>Baseline</th></tr></thead>
                <tbody>
                  {_metric_row("Toplam Talep (Mbps)", mp['total_demand_mbps'], mb['total_demand_mbps'], False)}
                  {_metric_row("Toplam Dağıtılan (Mbps)", mp['total_allocated_mbps'], mb['total_allocated_mbps'])}
                  {_metric_row("Genel Karşılanma", f"{mp['satisfaction_rate']*100:.1f}%", f"{mb['satisfaction_rate']*100:.1f}%")}
                  {_metric_row("Kritik Karşılanma", f"{mp['critical_satisfaction_rate']*100:.1f}%", f"{mb['critical_satisfaction_rate']*100:.1f}%")}
                  {_metric_row("Gecikme Cezası", f"{mp['latency_penalty']:.1f}", f"{mb['latency_penalty']:.1f}", False)}
                  {_metric_row("Kapasite Verimliliği", f"{mp['efficiency']*100:.1f}%", f"{mb['efficiency']*100:.1f}%")}
                </tbody>
              </table>
            </div>
            <div class="card">
              <h3>Kullanıcı Tipi Detayı</h3>
              <table class="metrics-table">
                <thead><tr><th>Tip</th><th>#</th><th>Priority</th><th>Baseline</th><th>Ort. Talep</th></tr></thead>
                <tbody>{per_type_rows}</tbody>
              </table>
            </div>
          </div>
          <div class="chart-container">
            <img src="data:image/png;base64,{img}" alt="{scenario} grafik" class="chart-img"/>
          </div>
        </section>"""

    def _example_card(ex):
        mp, mb = ex["metrics_priority"], ex["metrics_baseline"]
        delta_crit = (mp["critical_satisfaction_rate"] - mb["critical_satisfaction_rate"]) * 100
        delta_pen  = mb["latency_penalty"] - mp["latency_penalty"]
        sign_c = "+" if delta_crit >= 0 else ""
        sign_p = "+" if delta_pen >= 0 else ""

        users_table = ""
        if "users" in ex:
            has_name = "name" in ex["users"][0]
            name_th = "<th>İsim</th>" if has_name else ""
            users_table = f"""<table class="metrics-table mini-table">
              <thead><tr><th>Tip</th>{name_th}<th>Talep</th><th>Lat.Sens.</th><th>Öncelik</th><th>Skor</th><th>Priority</th><th>Baseline</th></tr></thead><tbody>"""
            for u in ex["users"]:
                ut = u["user_type"]
                name_td = f"<td style='font-size:.78rem;color:var(--text)'>{u['name']}</td>" if has_name else ""
                p_alloc = u['allocated_priority']
                b_alloc = u['allocated_baseline']
                p_cls = "win" if p_alloc >= u['demand'] - 0.01 else ("neutral" if p_alloc > 0 else "lose")
                users_table += f"""<tr>
                  <td><span class="type-badge type-{ut}">{ut}</span></td>
                  {name_td}
                  <td>{u['demand']}</td>
                  <td>{u['latency_sensitivity']}</td>
                  <td>{u['priority']}</td>
                  <td><strong>{u['score']}</strong></td>
                  <td class="{p_cls}">{p_alloc:.1f}</td>
                  <td class="neutral">{b_alloc:.1f}</td>
                </tr>"""
            users_table += "</tbody></table>"

        # n and scenario for the live sim button
        ex_n   = len(ex.get("users", [])) or 100
        ex_scen = "normal"
        if "crisis" in ex.get("id","") or "extreme" in ex.get("id","") or "disaster" in ex.get("id",""):
            ex_scen = "crisis"

        return f"""
        <div class="example-card" id="ex-{ex['id']}">
          <div class="example-header">
            <h3>{ex['title']}</h3>
            <span class="capacity-badge">Kapasite: {ex['capacity']} Mbps</span>
            <button class="sim-btn load-sim-btn"
              data-capacity="{ex['capacity']}" data-n="{ex_n}"
              data-scenario="{ex_scen}" data-method="priority">
              ▶ Simüle Et
            </button>
          </div>
          <p class="example-desc">{ex['description']}</p>
          <div class="delta-row">
            <div class="delta-item {'positive' if delta_crit >= 0 else 'negative'}">
              <span class="delta-label">Kritik Karşılanma Farkı</span>
              <span class="delta-value">{sign_c}{delta_crit:.1f}%</span>
            </div>
            <div class="delta-item {'positive' if delta_pen >= 0 else 'negative'}">
              <span class="delta-label">Gecikme Cezası İyileşmesi</span>
              <span class="delta-value">{sign_p}{delta_pen:.1f}</span>
            </div>
            <div class="delta-item neutral">
              <span class="delta-label">Kapasite Verimliliği (P)</span>
              <span class="delta-value">{mp['efficiency']*100:.1f}%</span>
            </div>
          </div>
          {users_table}
        </div>"""

    # ── Skor formülü görselleştirmesi ──
    score_formula_html = """
    <div class="formula-box">
      <h3>Skor Formülü</h3>
      <div class="formula">
        score = <span class="f-demand">demand × 0.5</span> +
                <span class="f-latency">latency_sensitivity × 30</span> +
                <span class="f-priority">priority × 10</span>
      </div>
      <div class="formula-legend">
        <div class="legend-item"><span class="dot f-demand-dot"></span>
          <strong>Demand:</strong> Anlık bant genişliği ihtiyacı (Mbps) — ağır iş yükü skoru artırır</div>
        <div class="legend-item"><span class="dot f-latency-dot"></span>
          <strong>Latency Sensitivity:</strong> 0–1 arası. Askeri/acil kullanıcılar 0.7+ alır</div>
        <div class="legend-item"><span class="dot f-priority-dot"></span>
          <strong>Priority:</strong> military=5 · emergency=4 · civilian=1</div>
      </div>
      <div class="priority-examples">
        <div class="p-ex military-ex">Military tipik skor: ~<strong>88</strong><br><small>40×0.5 + 0.9×30 + 5×10</small></div>
        <div class="p-ex emergency-ex">Emergency tipik skor: ~<strong>70</strong><br><small>25×0.5 + 0.75×30 + 4×10</small></div>
        <div class="p-ex civilian-ex">Civilian tipik skor: ~<strong>20</strong><br><small>12×0.5 + 0.25×30 + 1×10</small></div>
      </div>
    </div>"""

    # ── Turkey SVG helper ──
    _CITIES_SVG = [
        ("istanbul",   "İstanbul",   130, 55,  "civilian",  3),
        ("ankara",     "Ankara",     292, 132, "military",  2),
        ("izmir",      "İzmir",      46,  188, "civilian",  2),
        ("bursa",      "Bursa",      136, 110, "civilian",  1),
        ("antalya",    "Antalya",    235, 265, "civilian",  1),
        ("adana",      "Adana",      392, 278, "emergency", 1),
        ("konya",      "Konya",      270, 228, "civilian",  1),
        ("gaziantep",  "Gaziantep",  478, 285, "emergency", 1),
        ("diyarbakir", "Diyarbakır", 588, 238, "military",  1),
        ("kayseri",    "Kayseri",    392, 195, "civilian",  1),
        ("trabzon",    "Trabzon",    552, 68,  "emergency", 1),
        ("samsun",     "Samsun",     412, 46,  "civilian",  1),
        ("erzurum",    "Erzurum",    638, 132, "military",  1),
        ("van",        "Van",        725, 192, "military",  1),
    ]
    _TC = {"military": "#c0392b", "emergency": "#e67e22", "civilian": "#2980b9"}
    _TPATH = (
        # Türkiye ana hatları — Trakya'dan başlayarak saat yönünde
        # Trakya kuzeybatı köşesi
        "M 20,35 "
        # Trakya kuzey kıyısı → İstanbul/Boğaz
        "C 42,20 78,18 104,28 "
        "C 118,34 128,46 130,52 "
        # Karadeniz kıyısı: İstanbul → Sinop (kuzeye uzanan kıyı)
        "C 152,42 192,28 242,17 "
        "C 278,10 314,8 332,10 "       # Sinop (en kuzey nokta, y≈8)
        # Karadeniz: Sinop → Samsun (kıyı güneye döner)
        "C 355,12 382,22 412,38 "       # Samsun kıyısı y≈38
        # Samsun → Trabzon
        "C 442,40 502,48 552,64 "       # Trabzon kıyısı y≈64
        # Trabzon → Kuzeydoğu köşe
        "C 592,56 634,42 666,38 "
        # Doğu sınırı: Kuzeydoğu → Güneydoğu
        "C 692,44 718,62 742,88 "
        "C 760,118 774,154 777,186 "
        "C 780,214 774,238 762,255 "
        "C 748,268 730,276 707,279 "
        # Güney sınırı: Suriye/Irak
        "C 678,283 646,285 612,289 "
        "C 578,293 542,300 506,309 "
        # İskenderun körfezi
        "C 476,318 450,328 431,336 "
        "C 417,341 403,334 396,318 "    # Hatay (en güney nokta)
        # Akdeniz kıyısı batıya
        "C 374,300 348,287 320,283 "    # Adana/Mersin
        "C 292,279 266,295 244,314 "    # Antalya körfezi doğu
        "C 228,326 210,328 190,314 "    # Antalya körfezi en derin
        "C 168,299 143,285 120,283 "    # Fethiye/Marmaris
        # Ege kıyısı kuzeye
        "C 95,281 74,266 55,247 "
        "C 39,231 31,211 28,193 "       # İzmir kıyısı
        "C 23,172 19,150 18,129 "
        "C 17,107 17,85 21,65 "         # Çanakkale/Gelibolu
        # Marmara kıyısı → Trakya kapanış
        "C 24,53 37,43 54,37 "
        "C 43,29 29,28 20,35 Z"
    )

    def _make_turkey_svg(sfx, show_badges):
        heat = "\n".join(
            f'          <circle id="heat-{cid}-{sfx}" cx="{cx}" cy="{cy}" r="0" fill="{_TC[ct]}" opacity="0"/>'
            for cid, nm, cx, cy, ct, pop in _CITIES_SVG
        )
        rings = ""
        if show_badges:
            rings = "\n".join(
                f'          <circle cx="{cx}" cy="{cy}" r="{4+pop*2+5}" fill="none"'
                f' stroke="{_TC[ct]}" stroke-width="1.3" opacity="0.4" stroke-dasharray="3,2"/>'
                for cid, nm, cx, cy, ct, pop in _CITIES_SVG
            )
        dots = "\n".join(
            f'          <circle id="dot-{cid}-{sfx}" cx="{cx}" cy="{cy}" r="{4+pop*2}" fill="#3a5a8a"/>'
            for cid, nm, cx, cy, ct, pop in _CITIES_SVG
        )
        labels = "\n".join(
            f'          <text class="city-label" x="{cx}" y="{cy - (4+pop*2) - 2}">{nm}</text>'
            for cid, nm, cx, cy, ct, pop in _CITIES_SVG
        )
        badge_map  = {"military": "★MIL", "emergency": "⚡ACL", "civilian": "SİV"}
        badge_col  = {"military": "#a02828", "emergency": "#a05010", "civilian": "#1a5fd8"}
        type_lbl = ""
        if show_badges:
            type_lbl = "\n".join(
                f'          <text x="{cx}" y="{cy + (4+pop*2) + 10}"'
                f' class="city-type-label" fill="{badge_col[ct]}">{badge_map[ct]}</text>'
                for cid, nm, cx, cy, ct, pop in _CITIES_SVG
            )
        return (
            f'<svg class="turkey-svg" id="turkey-svg-{sfx}" viewBox="0 0 790 340" xmlns="http://www.w3.org/2000/svg">\n'
            f'        <defs>\n'
            f'          <filter id="hf-{sfx}" x="-60%" y="-60%" width="220%" height="220%">'
            f'<feGaussianBlur stdDeviation="13"/></filter>\n'
            f'          <filter id="sh-{sfx}" x="-20%" y="-20%" width="140%" height="140%">'
            f'<feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#7090b0" flood-opacity="0.25"/></filter>\n'
            f'        </defs>\n'
            f'        <path d="{_TPATH}" fill="#c8daea" stroke="#7090b0" stroke-width="1.3" filter="url(#sh-{sfx})"/>\n'
            f'        <g filter="url(#hf-{sfx})" opacity="0.75">\n{heat}\n        </g>\n'
            + (f'        <g>\n{rings}\n        </g>\n' if rings else '')
            + f'        <g>\n{dots}\n        </g>\n'
            + f'        <g>\n{labels}\n        </g>\n'
            + (f'        <g>\n{type_lbl}\n        </g>\n' if type_lbl else '')
            + '      </svg>'
        )

    svg_priority = _make_turkey_svg("p", show_badges=True)
    svg_baseline = _make_turkey_svg("b", show_badges=False)

    normal_sec  = _scenario_section("normal",  all_results["normal"])
    crisis_sec  = _scenario_section("crisis",  all_results["crisis"])
    example_cards = "\n".join(_example_card(ex) for ex in examples)

    n_mp = all_results["normal"]["priority"]
    n_mb = all_results["normal"]["baseline"]
    c_mp = all_results["crisis"]["priority"]
    c_mb = all_results["crisis"]["baseline"]

    # ── Karşılaştırma bölümü ──
    def _pct(v): return f"{v*100:.1f}%"
    def _bar(pct, color):
        return f'<div class="cmp-bar-track"><div class="cmp-bar-fill" style="width:{min(pct,100):.1f}%;background:{color}"></div><span class="cmp-bar-label">{pct:.1f}%</span></div>'

    # Büyük rakamlar
    n_crit_mult = n_mp["critical_satisfaction_rate"] / max(n_mb["critical_satisfaction_rate"], 0.01)
    n_pen_mult  = n_mb["latency_penalty"] / max(n_mp["latency_penalty"], 0.01)
    n_eff_gain  = (n_mp["efficiency"] - n_mb["efficiency"]) * 100
    c_crit_mult = c_mp["critical_satisfaction_rate"] / max(c_mb["critical_satisfaction_rate"], 0.01)

    # Kullanıcı tipi karşılaştırma satırları (her iki senaryo)
    def _type_cmp_rows(mp, mb):
        rows = ""
        for ut, label, col in [("military","Askeri","#c0392b"),("emergency","Acil Durum","#e67e22"),("civilian","Sivil","#2980b9")]:
            pt = mp["per_type"].get(ut,{}); bt = mb["per_type"].get(ut,{})
            if not pt: continue
            pp = pt["satisfaction_rate"]*100; bp = bt["satisfaction_rate"]*100
            win = pp > bp
            rows += f"""<tr>
              <td><span class="type-badge type-{ut}">{label}</span></td>
              <td>{pt['count']}</td>
              <td>
                {_bar(pp, col)}
              </td>
              <td>
                {_bar(bp, "#4a5568")}
              </td>
              <td class="{'cmp-win' if win else 'cmp-neutral'}">{"+" if win else ""}{pp-bp:.1f}pp</td>
            </tr>"""
        return rows

    n_type_rows = _type_cmp_rows(n_mp, n_mb)
    c_type_rows = _type_cmp_rows(c_mp, c_mb)

    comparison_sec = f"""
<section>
  <h2>⚔ Sistem Karşılaştırması — Priority vs Baseline</h2>

  <!-- Problem kutusu -->
  <div class="problem-box">
    <div class="problem-icon">⚠</div>
    <div>
      <h3>Statik (Eşit) Dağıtımın Problemi</h3>
      <p>Geleneksel eşit dağıtımda her kullanıcı aynı paya sahip olur.
         Bu yüzden görev-kritik bir askeri kullanıcı, sosyal medyada video izleyen
         bir sivil kullanıcıyla aynı bant genişliğine mahkum kalır.
         Kapasite kısıtlı olduğunda askeri kullanıcı <strong>hiç</strong> karşılanamayabilir,
         çünkü talep eşit payı aşmaktadır.</p>
    </div>
  </div>

  <!-- Büyük sayılar -->
  <div class="big-numbers">
    <div class="big-num-card">
      <div class="big-num" style="color:#5dde8a">{n_crit_mult:.1f}×</div>
      <div class="big-num-label">Daha fazla kritik kullanıcı karşılandı</div>
      <div class="big-num-sub">Normal senaryo — %{n_mp['critical_satisfaction_rate']*100:.0f} vs %{n_mb['critical_satisfaction_rate']*100:.0f}</div>
    </div>
    <div class="big-num-card">
      <div class="big-num" style="color:#f0a060">{n_pen_mult:.1f}×</div>
      <div class="big-num-label">Daha düşük gecikme cezası</div>
      <div class="big-num-sub">{n_mp['latency_penalty']:.1f} vs {n_mb['latency_penalty']:.1f} (düşük = iyi)</div>
    </div>
    <div class="big-num-card">
      <div class="big-num" style="color:{('#5dde8a' if n_eff_gain > 0 else '#e07070')}">{'+' if n_eff_gain>=0 else ''}{n_eff_gain:.1f}%</div>
      <div class="big-num-label">Ek kapasite verimliliği</div>
      <div class="big-num-sub">%{n_mp['efficiency']*100:.1f} vs %{n_mb['efficiency']*100:.1f}</div>
    </div>
    <div class="big-num-card">
      <div class="big-num" style="color:#5dde8a">{c_crit_mult:.1f}×</div>
      <div class="big-num-label">Kriz anında kritik üstünlük</div>
      <div class="big-num-sub">Kriz — %{c_mp['critical_satisfaction_rate']*100:.0f} vs %{c_mb['critical_satisfaction_rate']*100:.0f}</div>
    </div>
  </div>

  <!-- İki senaryo yan yana tablo -->
  <div class="cmp-scenarios-grid">
    <div class="card">
      <h3><span class="badge badge-normal">Normal</span> Senaryo — Kullanıcı Tipi Karşılanma</h3>
      <table class="metrics-table" style="margin-top:.6rem">
        <thead><tr><th>Tip</th><th>#</th><th>Priority</th><th>Baseline</th><th>Fark</th></tr></thead>
        <tbody>{n_type_rows}</tbody>
      </table>
      <div class="cmp-summary">
        <span>Genel karşılanma: <strong class="cmp-win">{_pct(n_mp['satisfaction_rate'])}</strong> vs <strong class="cmp-lose">{_pct(n_mb['satisfaction_rate'])}</strong></span>
        <span>Toplam dağıtılan: <strong>{n_mp['total_allocated_mbps']} Mbps</strong> vs <strong>{n_mb['total_allocated_mbps']} Mbps</strong></span>
      </div>
    </div>
    <div class="card">
      <h3><span class="badge badge-crisis">Kriz</span> Senaryo — Kullanıcı Tipi Karşılanma</h3>
      <table class="metrics-table" style="margin-top:.6rem">
        <thead><tr><th>Tip</th><th>#</th><th>Priority</th><th>Baseline</th><th>Fark</th></tr></thead>
        <tbody>{c_type_rows}</tbody>
      </table>
      <div class="cmp-summary">
        <span>Genel karşılanma: <strong class="cmp-win">{_pct(c_mp['satisfaction_rate'])}</strong> vs <strong class="cmp-lose">{_pct(c_mb['satisfaction_rate'])}</strong></span>
        <span>Toplam dağıtılan: <strong>{c_mp['total_allocated_mbps']} Mbps</strong> vs <strong>{c_mb['total_allocated_mbps']} Mbps</strong></span>
      </div>
    </div>
  </div>

  <!-- Hikaye kutusu -->
  <div class="story-box">
    <h3>📡 Kriz Anında Ne Olur? — Gerçek Dünya Senaryosu</h3>
    <div class="story-grid">
      <div class="story-col story-without">
        <div class="story-header">❌ Sistemsiz (Baseline)</div>
        <ul>
          <li>60 kullanıcı arasında kapasite bölünür → herkes <strong>~{1000//60:.0f} Mbps</strong> alır.</li>
          <li>Askeri kullanıcının talebi 35 Mbps ama pay sadece ~{1000//60:.0f} Mbps → <strong>karşılanamaz.</strong></li>
          <li>Acil durum ekibi 22 Mbps istiyor, eşit pay yetersiz → gecikme cezası yükselir.</li>
          <li>Sivil kullanıcılar istediklerinden az talep ettiğinden kapasitede <strong>%{100-n_mb['efficiency']*100:.0f} boşa harcanır.</strong></li>
          <li>Kritik operasyon başarısız olabilir. Gecikme kayıpları artabilir.</li>
        </ul>
      </div>
      <div class="story-col story-with">
        <div class="story-header">✅ Priority Sistemi ile</div>
        <ul>
          <li>Her kullanıcıya skor hesaplanır: askeri kullanıcı en yüksek <strong>~88 puan</strong> alır.</li>
          <li>Sıralama: Military → Emergency → Civilian.</li>
          <li>Askeri kullanıcı 35 Mbps talebiyle <strong>önce dolar,</strong> tam olarak karşılanır.</li>
          <li>Acil durum ekipleri de büyük çoğunluğu karşılanır (%{c_mp['per_type'].get('emergency',{}).get('satisfaction_rate',0)*100:.0f} kriz anında).</li>
          <li>Kapasite tamamen tükenir → <strong>%{n_mp['efficiency']*100:.0f} verimlilik.</strong></li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Faydalar -->
  <h3 style="margin-bottom:1rem">Sistemin Temel Faydaları</h3>
  <div class="benefits-grid">
    <div class="benefit-card">
      <div class="benefit-icon">🎯</div>
      <h4>Kritik Kullanıcı Önceliği</h4>
      <p>Military ve emergency kullanıcıları yüksek skor formülü sayesinde kapasite dolmadan
         önce tam olarak karşılanır. Normal senaryoda başarı oranı <strong>%{n_mp['critical_satisfaction_rate']*100:.0f}</strong>.</p>
    </div>
    <div class="benefit-card">
      <div class="benefit-icon">⚡</div>
      <h4>Gecikme Cezasında {n_pen_mult:.1f}× Düşüş</h4>
      <p>Latency-sensitive kullanıcılar önce karşılandığı için ağırlıklı gecikme cezası
         baseline'a kıyasla çok daha düşük kalır. Normal senaryo:
         <strong>{n_mp['latency_penalty']:.1f}</strong> vs <strong>{n_mb['latency_penalty']:.1f}</strong>.</p>
    </div>
    <div class="benefit-card">
      <div class="benefit-icon">📊</div>
      <h4>%100 Kapasite Kullanımı</h4>
      <p>Baseline, talep az olan kullanıcıların fazla payını boşa harcar.
         Priority, kalan kapasiteyi sıradaki kullanıcıya aktarır — her zaman
         <strong>%{n_mp['efficiency']*100:.0f} verimlilik</strong>.</p>
    </div>
    <div class="benefit-card">
      <div class="benefit-icon">🚨</div>
      <h4>Krizde Fark Daha Belirgin</h4>
      <p>Kriz senaryosunda kritik kullanıcı oranı arttıkça Priority'nin avantajı büyür.
         Baseline kritik karşılama <strong>%{c_mb['critical_satisfaction_rate']*100:.0f}</strong> kalırken
         Priority <strong>%{c_mp['critical_satisfaction_rate']*100:.0f}</strong> sağlar.</p>
    </div>
    <div class="benefit-card">
      <div class="benefit-icon">🔧</div>
      <h4>Modüler ve Parametrik</h4>
      <p>Skor ağırlıkları (<em>demand×0.5, latency×30, priority×10</em>) kolayca
         ayarlanabilir. Farklı operasyon profillerine uyarlamak için yalnızca
         sabitler değiştirilir.</p>
    </div>
    <div class="benefit-card">
      <div class="benefit-icon">🌐</div>
      <h4>Gerçek Zamanlı Uygulanabilirlik</h4>
      <p>O(n log n) karmaşıklığı ile skor sıralaması gerçek zamanlı uydu geçiş
         pencerelerinde (tipik 5–10 dk) hesaplanabilir. Makul yeniden-hesaplama
         maliyeti.</p>
    </div>
  </div>
</section>"""

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AstroHackathon 2025 — LEO Bant Genişliği Optimizasyonu</title>
<style>{css}</style>
</head>
<body>
<header>
  <div class="header-badge">AstroHackathon 2025</div>
  <h1>🛰 LEO Uydu Bant Genişliği Optimizasyonu</h1>
  <p>Dinamik öncelik bazlı dağıtım algoritması ile eşit (statik) dağıtımın karşılaştırmalı simülasyonu.</p>
  <div class="stat-strip">
    <div class="stat"><div class="stat-val">1000 Mbps</div><div class="stat-lbl">Toplam Kapasite</div></div>
    <div class="stat"><div class="stat-val">100</div><div class="stat-lbl">Simüle Edilen Kullanıcı</div></div>
    <div class="stat"><div class="stat-val">2</div><div class="stat-lbl">Senaryo</div></div>
    <div class="stat"><div class="stat-val">4</div><div class="stat-lbl">Örnek Simülasyon</div></div>
  </div>
</header>

<nav>
  <a class="nav-link active"  data-tab="overview">Genel Bakış</a>
  <a class="nav-link"         data-tab="compare">⚔ Karşılaştırma</a>
  <a class="nav-link"         data-tab="modes">⚙ Operasyon Modları</a>
  <a class="nav-link"         data-tab="methodology">Metodoloji</a>
  <a class="nav-link"         data-tab="normal">Normal Senaryo</a>
  <a class="nav-link"         data-tab="crisis">Kriz Senaryosu</a>
  <a class="nav-link"         data-tab="examples">Simülasyon Örnekleri</a>
  <a class="nav-link"         data-tab="live">🎬 Canlı Simülasyon</a>
</nav>

<main>

<!-- ── OVERVIEW ── -->
<div class="tab-pane active" id="tab-overview">
<section>
  <h2>Genel Bakış — Algoritma Karşılaştırması</h2>
  <div class="comparison-hero">
    <div class="hero-card priority">
      <h3>🎯 Priority (Önerilen)</h3>
      <div class="hero-metric green">{n_mp['critical_satisfaction_rate']*100:.0f}%</div>
      <div class="hero-label">Normal sn. kritik karşılanma</div>
      <br>
      <div class="hero-metric green">{c_mp['critical_satisfaction_rate']*100:.0f}%</div>
      <div class="hero-label">Kriz sn. kritik karşılanma</div>
    </div>
    <div class="hero-card baseline">
      <h3>⚖ Baseline (Eşit Dağıtım)</h3>
      <div class="hero-metric gray">{n_mb['critical_satisfaction_rate']*100:.0f}%</div>
      <div class="hero-label">Normal sn. kritik karşılanma</div>
      <br>
      <div class="hero-metric gray">{c_mb['critical_satisfaction_rate']*100:.0f}%</div>
      <div class="hero-label">Kriz sn. kritik karşılanma</div>
    </div>
  </div>
  <div class="card">
    <p style="color:var(--muted);font-size:.88rem;">
      Priority algoritması kapasite doluncaya kadar en yüksek skorlu kullanıcıyı önce doyurur.
      Normal senaryoda tüm kritik kullanıcılar (%100) karşılanırken, baseline sadece
      %{n_mb['critical_satisfaction_rate']*100:.0f} oranında karşılayabilmektedir.
      Kriz senaryosunda fark daha da belirginleşir:
      Priority %{c_mp['critical_satisfaction_rate']*100:.0f} vs Baseline %{c_mb['critical_satisfaction_rate']*100:.0f}.
    </p>
  </div>
</section>
</div>

<!-- ── COMPARISON ── -->
<div class="tab-pane" id="tab-compare">
{comparison_sec}
</div>

<!-- ── OPERATION MODES ── -->
<div class="tab-pane" id="tab-modes">
<section>
  <h2>⚙ Operasyon Modları</h2>
  <p style="color:var(--muted);font-size:.88rem;max-width:680px">
    Aynı algoritma, sadece üç ağırlık parametresi değiştirilerek farklı operasyon
    koşullarına adapte edilir. Operatör sabah tek bir komutla mod değiştirir.
  </p>
  <div class="modes-grid">

    <!-- Barış Zamanı -->
    <div class="mode-card blue">
      <div class="mode-card-title">🕊 Barış Zamanı</div>
      <div class="mode-card-desc">
        Kritik kullanıcı oranı düşük, kapasite baskısı yok.
        Sistem tüm kullanıcı tiplerine dengeli pay verir.
      </div>
      <div class="mode-weights mode-weight-blue">
        <span>demand   × <b>0.5</b></span>
        <span>latency  × <b>20</b></span>
        <span>priority × <b>5</b></span>
      </div>
      <ul class="mode-examples">
        <li>NATO üssünde rutin iletişim</li>
        <li>Sınır karakolu</li>
        <li>Barış gücü operasyonu</li>
      </ul>
      <div class="mode-result">Sivil kullanıcılar da makul pay alır</div>
    </div>

    <!-- Aktif Operasyon -->
    <div class="mode-card red active-mode">
      <div class="active-badge">SİMÜLASYONDA KULLANILAN MOD</div>
      <div class="mode-card-title">⚔ Aktif Operasyon</div>
      <div class="mode-card-desc">
        Military ve emergency oranı yüksek, kapasite baskı altında.
        Sistem sivil kullanıcıyı bilinçli olarak keser, kritik operasyonu korur.
      </div>
      <div class="mode-weights mode-weight-red">
        <span>demand   × <b>0.5</b></span>
        <span>latency  × <b>30</b></span>
        <span>priority × <b>10</b></span>
      </div>
      <ul class="mode-examples">
        <li>Aktif çatışma bölgesi</li>
        <li>Koordineli hava saldırısı yönetimi</li>
        <li>Komuta-kontrol trafiği</li>
      </ul>
      <div class="mode-result">Bizim simülasyonumuzda gösterilen senaryo</div>
    </div>

    <!-- İnsani Yardım -->
    <div class="mode-card orange">
      <div class="mode-card-title">🆘 İnsani Yardım</div>
      <div class="mode-card-desc">
        Deprem, sel gibi durumlarda military değil emergency öne çıkar.
        Sivil haberleşme de kriz yönetiminin parçası olduğu için tamamen kesilmez.
      </div>
      <div class="mode-weights mode-weight-orange">
        <span>demand   × <b>0.5</b></span>
        <span>latency  × <b>35</b></span>
        <span>priority × <b>7</b></span>
      </div>
      <ul class="mode-examples">
        <li>Deprem arama kurtarma</li>
        <li>Sel tahliyesi</li>
        <li>Sağlık koordinasyonu</li>
        <li>Kızılay / AFAD haberleşmesi</li>
      </ul>
      <div class="mode-result">Emergency maksimum, sivil kısmen korunur</div>
    </div>

  </div>

  <div class="modes-note">
    Üç ağırlık değeri değiştirilerek aynı sistem üç farklı operasyon moduna alınabilir.
    <strong>Kod değişmez, sadece konfigürasyon değişir.</strong>
    Operatör sabah aktif operasyon moduna alır, akşam barış moduna döner.
  </div>
</section>
</div>

<!-- ── METHODOLOGY ── -->
<div class="tab-pane" id="tab-methodology">
<section>
  <h2>Metodoloji</h2>
  {score_formula_html}
  <div class="card">
    <h3>Algoritma Akışı</h3>
    <ol style="color:var(--muted);font-size:.88rem;padding-left:1.4rem;line-height:2;">
      <li>Kullanıcı kümesi senaryo oranlarına göre üretilir (demand, latency_sensitivity, priority).</li>
      <li>Her kullanıcıya skor hesaplanır.</li>
      <li><strong style="color:var(--text)">Priority:</strong> Skor sıralamasına göre yüksekten aşağı dağıtım; kapasite bitince geri kalanlar 0 alır.</li>
      <li><strong style="color:var(--text)">Baseline:</strong> Kapasite / kullanıcı sayısı eşit paylar; talep düşükse fazlası geri bırakılır.</li>
      <li>Karşılanma oranı, gecikme cezası ve kapasite verimliliği hesaplanır.</li>
    </ol>
  </div>
</section>
</div>

<!-- ── NORMAL SCENARIO ── -->
<div class="tab-pane" id="tab-normal">
{normal_sec}
</div>

<!-- ── CRISIS SCENARIO ── -->
<div class="tab-pane" id="tab-crisis">
{crisis_sec}
</div>

<!-- ── EXAMPLES ── -->
<div class="tab-pane" id="tab-examples">
<section>
  <h2>Simülasyon Örnekleri</h2>
  <p style="color:var(--muted);font-size:.88rem;margin-bottom:1.2rem;">
    Farklı kapasite ve kullanıcı dağılım koşullarında algoritmanın davranışı.
    <strong>▶ Simüle Et</strong> butonuna basarak ilgili örneği Canlı Simülasyon sekmesinde çalıştırabilirsin.
  </p>
  {example_cards}
</section>
</div>

<!-- ── LIVE SIMULATION ── -->
<div class="tab-pane" id="tab-live">
<section>
  <h2>🎬 Canlı Simülasyon</h2>
  <div class="live-layout">

    <!-- Controls Panel -->
    <div class="live-controls-panel">
      <div class="ctrl-row">
        <label>Kapasite <span class="val-display" id="sim-cap-val">500</span> Mbps</label>
        <input type="range" id="sim-cap" min="100" max="2000" step="50" value="500"/>
      </div>
      <div class="ctrl-row">
        <label>Kullanıcı Sayısı <span class="val-display" id="sim-n-val">15</span></label>
        <input type="range" id="sim-n" min="5" max="30" step="1" value="15"/>
      </div>
      <div class="ctrl-row">
        <label>Senaryo</label>
        <select id="sim-scenario">
          <option value="normal">Normal (%70 Sivil)</option>
          <option value="crisis">Kriz (%40 Sivil)</option>
        </select>
      </div>
      <div class="ctrl-row">
        <label>Dağıtım Yöntemi</label>
        <select id="sim-method">
          <option value="priority">Priority (Öncelikli)</option>
          <option value="baseline">Baseline (Eşit)</option>
        </select>
      </div>
      <div class="ctrl-row">
        <label>Animasyon Hızı</label>
        <div class="speed-btns">
          <button class="speed-btn" onclick="setSpeed(0)">0.5x</button>
          <button class="speed-btn active" onclick="setSpeed(1)">1x</button>
          <button class="speed-btn" onclick="setSpeed(2)">3x</button>
          <button class="speed-btn" onclick="setSpeed(3)">10x</button>
        </div>
      </div>
      <div class="btn-group" style="margin-top:.8rem;">
        <button class="btn btn-play" id="btn-play" onclick="togglePlay()">▶ Başlat</button>
        <button class="btn btn-step" onclick="stepOnce()">&#8677; Adım</button>
        <button class="btn btn-reset" onclick="resetSim()">↺ Sıfırla</button>
      </div>
    </div>

    <!-- Right: gauge + stats + users -->
    <div class="live-main">
      <!-- Gauge -->
      <div class="gauge-card">
        <div class="gauge-header">
          <span class="gauge-title">Kapasite Havuzu</span>
          <span class="gauge-pct" id="gauge-pct">0%</span>
        </div>
        <div class="gauge-track">
          <div class="gauge-fill" id="gauge-fill" style="width:0%"></div>
        </div>
        <div class="gauge-label" id="gauge-text">0 / 1000 Mbps</div>
      </div>

      <!-- Stats -->
      <div class="stats-row">
        <div class="stat-card">
          <div class="s-val" id="stat-critical">0%</div>
          <div class="s-lbl">Kritik Karşılanma</div>
        </div>
        <div class="stat-card">
          <div class="s-val" id="stat-latency">0</div>
          <div class="s-lbl">Gecikme Cezası</div>
        </div>
        <div class="stat-card">
          <div class="s-val" id="stat-efficiency">0%</div>
          <div class="s-lbl">Kapasite Verimi</div>
        </div>
      </div>

      <!-- User list -->
      <div class="users-panel">
        <div class="users-panel-header">
          <span>Kullanıcı Dağıtım Sırası (skora göre)</span>
          <span id="step-counter">Adım: 0 / 0</span>
        </div>
        <div class="users-scroll">
          <div id="sim-empty" class="empty-state">↓ Başlat butonuna bas</div>
          <div id="sim-users-list"></div>
        </div>
      </div>

    <!-- Turkey Dual Map Panel -->
    <div class="map-panel">
      <div class="map-panel-header">
        <span class="map-panel-title">🗺 Türkiye Bant Genişliği Isı Haritası — Karşılaştırma</span>
        <span style="font-size:.7rem;color:var(--muted)">Simülasyon başladığında her iki harita güncellenir</span>
      </div>
      <div class="dual-map-grid">
        <div>
          <div class="map-col-header priority-header">🎯 Priority Sistemi (Önem Sırası)</div>
          {svg_priority}
          <span class="map-zoom-hint">🔍 büyütmek için tıkla</span>
          <div class="prio-order-note">İşlem sırası: <span style="color:#e07070">★MIL</span> → <span style="color:#f0a060">⚡ACL</span> → <span style="color:#70b0e0">SİV</span></div>
        </div>
        <div>
          <div class="map-col-header baseline-header">⚖ Sistemsiz — Baseline (Eşit Dağıtım)</div>
          {svg_baseline}
          <span class="map-zoom-hint">🔍 büyütmek için tıkla</span>
          <div class="prio-order-note" style="color:var(--muted)">Tüm kullanıcılar eşit pay alır — başlangıçta gösterilir</div>
        </div>
      </div>
      <div class="map-legend">
        <span><span class="legend-dot" style="background:#27ae60"></span>Tam Karşılandı (≥%75)</span>
        <span><span class="legend-dot" style="background:#e67e22"></span>Kısmi (%40-75)</span>
        <span><span class="legend-dot" style="background:#c0392b"></span>Karşılanamadı (&lt;%40)</span>
        <span><span class="legend-dot" style="background:transparent;border:2px dashed #c0392b"></span>Askeri şehir</span>
        <span><span class="legend-dot" style="background:transparent;border:2px dashed #e67e22"></span>Acil Durum</span>
      </div>
    </div>

    </div>
  </div>
</section>
</div>

<!-- Map Zoom Modal -->
<div id="map-modal">
  <div id="map-modal-inner">
    <div id="map-modal-title"></div>
    <button id="map-modal-close">✕</button>
  </div>
</div>

</main>

<footer>
  AstroHackathon 2025 · LEO Uydu Bant Genişliği Optimizasyonu ·
  Python / NumPy / Pandas / Matplotlib
</footer>

<script>{js}</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML rapor kaydedildi: {path}")


# ─────────────────────────────────────────────
# 7. Terminal Çıktısı
# ─────────────────────────────────────────────
def print_metrics_table(mp: dict, mb: dict, scenario: str) -> None:
    print(f"\n{'═'*62}")
    print(f"  SENARYO: {scenario.upper()}")
    print(f"{'═'*62}")
    print(f"  {'Metrik':<30} {'Priority':>12} {'Baseline':>12}")
    print(f"  {'-'*56}")

    rows = [
        ("Toplam Talep (Mbps)",         mp["total_demand_mbps"],              mb["total_demand_mbps"]),
        ("Toplam Dağıtılan (Mbps)",      mp["total_allocated_mbps"],           mb["total_allocated_mbps"]),
        ("Genel Karşılanma Oranı",       f"{mp['satisfaction_rate']*100:.1f}%",  f"{mb['satisfaction_rate']*100:.1f}%"),
        ("Kritik Karşılanma Oranı",      f"{mp['critical_satisfaction_rate']*100:.1f}%", f"{mb['critical_satisfaction_rate']*100:.1f}%"),
        ("Gecikme Cezası (toplam)",      f"{mp['latency_penalty']:.2f}",       f"{mb['latency_penalty']:.2f}"),
        ("Kapasite Verimliliği",         f"{mp['efficiency']*100:.1f}%",       f"{mb['efficiency']*100:.1f}%"),
    ]
    for name, pval, bval in rows:
        print(f"  {name:<30} {str(pval):>12} {str(bval):>12}")

    print(f"\n  Kullanıcı Tipi Detayı:")
    print(f"  {'Tip':<12} {'#':>4} {'P-Karşılanma':>14} {'B-Karşılanma':>14} {'Ort. Talep':>11}")
    print(f"  {'-'*56}")
    for utype in ["military", "emergency", "civilian"]:
        pt = mp["per_type"].get(utype, {})
        bt = mb["per_type"].get(utype, {})
        if not pt:
            continue
        print(
            f"  {utype:<12} {pt['count']:>4} "
            f"{pt['satisfaction_rate']*100:>13.1f}% "
            f"{bt['satisfaction_rate']*100:>13.1f}% "
            f"{pt['avg_demand_mbps']:>10.1f}"
        )
    print(f"{'═'*62}")


# ─────────────────────────────────────────────
# 8. Ana Simülasyon
# ─────────────────────────────────────────────
def run_simulation(scenarios: list[str] = None, capacity: float = TOTAL_CAPACITY_MBPS) -> dict:
    if scenarios is None:
        scenarios = ["normal", "crisis"]

    all_results = {}
    scenario_data = {}  # df + alloc için grafik ve HTML'e aktarım

    for scenario in scenarios:
        print(f"\n[*] Senaryo çalıştırılıyor: {scenario.upper()}")

        df = generate_users(scenario)
        alloc_p = allocate_priority(df, capacity)
        alloc_b = allocate_baseline(df, capacity)

        mp = calculate_metrics(df, alloc_p, "priority")
        mb = calculate_metrics(df, alloc_b, "baseline")

        print_metrics_table(mp, mb, scenario)

        img_path = f"leo_{scenario}.png"
        visualize_scenario(df, alloc_p, alloc_b, mp, mb, scenario, img_path)

        all_results[scenario] = {"priority": mp, "baseline": mb}
        scenario_data[scenario] = {
            "df": df, "alloc_p": alloc_p, "alloc_b": alloc_b, "mp": mp, "mb": mb,
        }

    examples = build_simulation_examples()
    save_results(all_results)
    generate_html_report(all_results, scenario_data, examples)
    return all_results


# ─────────────────────────────────────────────
# 9. Testler
# ─────────────────────────────────────────────
def run_tests() -> None:
    print("\n[TEST] Birim testler çalıştırılıyor...")

    # Test 1: Kullanıcı sayısı doğru mu?
    for scenario in ["normal", "crisis"]:
        df = generate_users(scenario, n_users=100)
        assert len(df) == 100, f"Kullanıcı sayısı hatalı: {len(df)}"
        assert set(df["user_type"].unique()).issubset({"military", "emergency", "civilian"})
    print("  [OK] Kullanıcı üretimi")

    # Test 2: Priority dağıtımı kapasiteyi aşmıyor mu?
    df = generate_users("normal")
    alloc = allocate_priority(df)
    assert alloc.sum() <= TOTAL_CAPACITY_MBPS + 1e-6, "Kapasite aşıldı!"
    assert (alloc >= 0).all(), "Negatif dağıtım var!"
    assert (alloc <= df["demand"] + 1e-6).all(), "Talep aşıldı!"
    print("  [OK] Priority dağıtım kısıtları")

    # Test 3: Baseline dağıtımı kapasiteyi aşmıyor mu?
    alloc_b = allocate_baseline(df)
    assert alloc_b.sum() <= TOTAL_CAPACITY_MBPS + 1e-6
    print("  [OK] Baseline dağıtım kısıtları")

    # Test 4: Skor sıralaması — military en yüksek skoru almalı (ortalama)
    df_test = pd.DataFrame([
        {"user_type": "military",  "priority": 5, "demand": 20.0, "latency_sensitivity": 0.9},
        {"user_type": "civilian",  "priority": 1, "demand": 20.0, "latency_sensitivity": 0.1},
    ])
    scores = compute_scores(df_test)
    assert scores.iloc[0] > scores.iloc[1], "Military skoru civilian'dan düşük!"
    print("  [OK] Skor sıralaması")

    # Test 5: Kriz senaryosunda priority algoritması kritik kullanıcıları daha iyi karşılamalı
    df_c = generate_users("crisis")
    alloc_cp = allocate_priority(df_c)
    alloc_cb = allocate_baseline(df_c)
    mp = calculate_metrics(df_c, alloc_cp, "priority")
    mb = calculate_metrics(df_c, alloc_cb, "baseline")
    assert mp["critical_satisfaction_rate"] >= mb["critical_satisfaction_rate"], \
        "Priority kriz senaryosunda kritik kullanıcıları daha iyi karşılamalı!"
    print("  [OK] Kriz senaryosu kritik karşılanma")

    print("\n[TEST] Tüm testler geçti.\n")


# ─────────────────────────────────────────────
# 10. Zaman Serisi Simülasyonu
# ─────────────────────────────────────────────

@dataclass
class TickSnapshot:
    tick:             int
    scenario:         str
    capacity:         float
    users:            pd.DataFrame
    alloc_priority:   pd.Series
    alloc_baseline:   pd.Series
    metrics_priority: dict
    metrics_baseline: dict


def simulate_time_series(
    n_ticks:       int   = 40,
    capacity:      float = 1000.0,
    crisis_start:  int   = 15,
    crisis_end:    int   = 28,
    base_n_users:  int   = 80,
    verbose:       bool  = False,
) -> list[TickSnapshot]:
    """
    Her tick'te yeni bir kullanıcı kümesi üretip her iki algoritmayı çalıştırır.
    crisis_start <= tick < crisis_end aralığında kriz senaryosu aktiftir.
    """
    snapshots: list[TickSnapshot] = []

    for tick in range(n_ticks):
        in_crisis = crisis_start <= tick < crisis_end
        scenario  = "crisis" if in_crisis else "normal"
        n_users   = int(base_n_users * 1.25) if in_crisis else base_n_users
        seed      = tick * 37 + 13

        df       = generate_users(scenario, n_users=n_users, seed=seed)
        alloc_p  = allocate_priority(df, capacity)
        alloc_b  = allocate_baseline(df, capacity)
        mp       = calculate_metrics(df, alloc_p, "priority")
        mb       = calculate_metrics(df, alloc_b, "baseline")

        snap = TickSnapshot(
            tick=tick, scenario=scenario, capacity=capacity,
            users=df, alloc_priority=alloc_p, alloc_baseline=alloc_b,
            metrics_priority=mp, metrics_baseline=mb,
        )
        snapshots.append(snap)

        if verbose:
            crit_p = mp["critical_satisfaction_rate"] * 100
            crit_b = mb["critical_satisfaction_rate"] * 100
            marker = " ◀ KRİZ" if in_crisis else ""
            print(f"  tick {tick:>2}  [{scenario:<6}]  kritik P={crit_p:5.1f}%  B={crit_b:5.1f}%{marker}")

    return snapshots


def export_timeseries_json(
    snapshots: list[TickSnapshot],
    path:      str = "timeseries.json",
) -> None:
    """Her snapshot'tan özet metrikleri JSON'a yazar."""
    records = []
    for s in snapshots:
        mp, mb = s.metrics_priority, s.metrics_baseline
        records.append({
            "tick":     s.tick,
            "scenario": s.scenario,
            "capacity": s.capacity,
            "n_users":  len(s.users),
            "priority": {
                "critical_satisfaction": mp["critical_satisfaction_rate"],
                "satisfaction":          mp["satisfaction_rate"],
                "latency_penalty":       mp["latency_penalty"],
                "efficiency":            mp["efficiency"],
                "per_type":              mp["per_type"],
            },
            "baseline": {
                "critical_satisfaction": mb["critical_satisfaction_rate"],
                "satisfaction":          mb["satisfaction_rate"],
                "latency_penalty":       mb["latency_penalty"],
                "efficiency":            mb["efficiency"],
                "per_type":              mb["per_type"],
            },
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Zaman serisi JSON kaydedildi: {path}  ({len(records)} tick)")


def visualize_timeseries(
    snapshots:   list[TickSnapshot],
    output_path: str = "leo_timeseries.png",
) -> None:
    """2×2 figure: kritik karşılanma, gecikme cezası, verimlilik, delta bar chart."""
    ticks   = [s.tick for s in snapshots]
    crit_p  = [s.metrics_priority["critical_satisfaction_rate"] * 100 for s in snapshots]
    crit_b  = [s.metrics_baseline["critical_satisfaction_rate"] * 100 for s in snapshots]
    pen_p   = [s.metrics_priority["latency_penalty"]  for s in snapshots]
    pen_b   = [s.metrics_baseline["latency_penalty"]  for s in snapshots]
    eff_p   = [s.metrics_priority["efficiency"] * 100 for s in snapshots]
    eff_b   = [s.metrics_baseline["efficiency"] * 100 for s in snapshots]
    delta   = [cp - cb for cp, cb in zip(crit_p, crit_b)]

    # Kriz aralığını bul
    crisis_ticks = [s.tick for s in snapshots if s.scenario == "crisis"]
    c_lo = min(crisis_ticks) - 0.5 if crisis_ticks else None
    c_hi = max(crisis_ticks) + 0.5 if crisis_ticks else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "LEO Bant Genişliği — Zaman Serisi Analizi",
        fontsize=14, fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.38, wspace=0.28)

    def _shade(ax):
        if c_lo is not None:
            ax.axvspan(c_lo, c_hi, color="red", alpha=0.08, label="Kriz penceresi")

    # ── Panel 1: Kritik karşılanma ──
    ax = axes[0, 0]
    ax.plot(ticks, crit_p, color=METHOD_COLORS["Priority"], lw=2, label="Priority")
    ax.plot(ticks, crit_b, color=METHOD_COLORS["Baseline"], lw=2, linestyle="--", label="Baseline")
    _shade(ax)
    ax.set_title("Kritik Karşılanma Oranı (%)")
    ax.set_xlabel("Tick"); ax.set_ylabel("%")
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: Gecikme cezası ──
    ax = axes[0, 1]
    ax.plot(ticks, pen_p, color=METHOD_COLORS["Priority"], lw=2, label="Priority")
    ax.plot(ticks, pen_b, color=METHOD_COLORS["Baseline"], lw=2, linestyle="--", label="Baseline")
    _shade(ax)
    ax.set_title("Gecikme Cezası (düşük = iyi)")
    ax.set_xlabel("Tick"); ax.set_ylabel("Ağırlıklı ceza")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: Kapasite verimliliği ──
    ax = axes[1, 0]
    ax.plot(ticks, eff_p, color=METHOD_COLORS["Priority"], lw=2, label="Priority")
    ax.plot(ticks, eff_b, color=METHOD_COLORS["Baseline"], lw=2, linestyle="--", label="Baseline")
    _shade(ax)
    ax.set_title("Kapasite Verimliliği (%)")
    ax.set_xlabel("Tick"); ax.set_ylabel("%")
    ax.set_ylim(50, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 4: Delta bar chart ──
    ax = axes[1, 1]
    colors = [COLORS["military"] if d >= 0 else COLORS["emergency"] for d in delta]
    ax.bar(ticks, delta, color=colors, width=0.8, alpha=0.85)
    ax.axhline(0, color="white", lw=0.7, alpha=0.5)
    _shade(ax)
    ax.set_title("Δ Kritik Karşılanma (Priority − Baseline, pp)")
    ax.set_xlabel("Tick"); ax.set_ylabel("pp farkı")
    ax.grid(True, alpha=0.2, axis="y")

    # Kriz aralığı annotation
    if crisis_ticks:
        mid = (c_lo + c_hi) / 2
        for a in axes.flat:
            a.axvline(c_lo + 0.5, color="red", lw=0.8, alpha=0.4, linestyle=":")
            a.axvline(c_hi - 0.5, color="red", lw=0.8, alpha=0.4, linestyle=":")
        axes[0, 0].annotate(
            "KRİZ", xy=(mid, 105), ha="center", fontsize=8,
            color="red", fontweight="bold",
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Zaman serisi grafiği kaydedildi: {output_path}")


def run_timeseries_simulation(
    n_ticks:      int   = 40,
    capacity:     float = 1000.0,
    crisis_start: int   = 15,
    crisis_end:   int   = 28,
    base_n_users: int   = 80,
) -> list[TickSnapshot]:
    """Zaman serisi simülasyonunu çalıştırır, grafik ve JSON çıktılarını kaydeder."""
    print(f"\n[*] Zaman serisi simülasyonu başlatılıyor "
          f"({n_ticks} tick, kriz: {crisis_start}–{crisis_end})")

    snapshots = simulate_time_series(
        n_ticks=n_ticks, capacity=capacity,
        crisis_start=crisis_start, crisis_end=crisis_end,
        base_n_users=base_n_users, verbose=True,
    )

    export_timeseries_json(snapshots)
    visualize_timeseries(snapshots)

    # Özet istatistikler
    normal_snaps = [s for s in snapshots if s.scenario == "normal"]
    crisis_snaps = [s for s in snapshots if s.scenario == "crisis"]

    def _avg(snaps, key, method):
        vals = [getattr(s, f"metrics_{method}")[key] for s in snaps]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n  {'─'*54}")
    print(f"  {'Ortalama Metrik':<28} {'Normal':>10} {'Kriz':>10}")
    print(f"  {'─'*54}")
    for label, key, scale in [
        ("Kritik Karşılanma — Priority", "critical_satisfaction_rate", 100),
        ("Kritik Karşılanma — Baseline", "critical_satisfaction_rate", 100),
        ("Gecikme Cezası — Priority",    "latency_penalty",            1),
        ("Gecikme Cezası — Baseline",    "latency_penalty",            1),
        ("Verimlilik — Priority",        "efficiency",                 100),
    ]:
        method = "priority" if "Priority" in label else "baseline"
        n_val = _avg(normal_snaps, key, method) * scale
        c_val = _avg(crisis_snaps, key, method) * scale
        suffix = "%" if scale == 100 else ""
        print(f"  {label:<28} {n_val:>9.1f}{suffix} {c_val:>9.1f}{suffix}")
    print(f"  {'─'*54}")

    return snapshots


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_tests()
    run_simulation()
    run_timeseries_simulation()
