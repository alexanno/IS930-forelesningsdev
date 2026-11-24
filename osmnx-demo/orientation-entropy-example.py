import logging
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ingen GUI nødvendig, vi lagrer til fil
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("entropi")

# Koordinater (lat, lon) for sentrum i hver by
cities = {
    "Kristiansand": (58.1467, 7.9956),  # Kvadraturen
    "Trondheim": (63.4305, 10.3951),  # Midtbyen
    "Lillehammer": (61.1153, 10.4662),  # Storgata / Sentrum
    "Oslo": (59.9139, 10.7522),  # Sentrum
    "Paris": (48.8566, 2.3522),  # Île de la Cité / Sentrum
    "Roma": (41.9028, 12.4964),  # Centro Storico
}

RADIUS = 1000  # meter
NUM_BINS = 36  # 10 grader per sektor
OUT_DIR = Path(__file__).parent / "figurer"
OUT_DIR.mkdir(exist_ok=True)

H_GRID = math.log(4)  # ~1.386 (perfekt rutenett)
H_MAX = math.log(NUM_BINS)  # ~3.584 (maksimal entropi)


def fetch_graph(city, coords, retries=4):
    """Hent gatenett med eksponentiell backoff (Overpass svarer av og til 502)."""
    for attempt in range(1, retries + 1):
        try:
            return ox.graph_from_point(coords, dist=RADIUS, network_type="drive")
        except Exception as e:
            if attempt == retries:
                raise
            wait = 10 * 2 ** (attempt - 1)
            log.warning("%s: forsøk %d/%d feilet (%s), prøver igjen om %d s",
                        city, attempt, retries, type(e).__name__, wait)
            time.sleep(wait)


def bearing_histogram(Gu, num_bins=NUM_BINS):
    """Vektet histogram over gateretninger, sentrert slik at N ligger midt i første sektor."""
    bearings = np.array([d["bearing"] for _, _, d in Gu.edges(data=True) if "bearing" in d])
    bin_edges = np.arange(num_bins + 1) * 360 / num_bins
    split = np.hstack([bearings, (bearings + 180) % 360])  # begge retninger
    shifted = (split - 360 / num_bins / 2) % 360
    counts, _ = np.histogram(shifted, bins=bin_edges)
    return counts, bin_edges


def polar_plot(ax, counts, title):
    width = 2 * np.pi / len(counts)
    theta = np.arange(len(counts)) * width
    ax.bar(theta, counts / counts.sum(), width=width, bottom=0.0,
           color="#1f77b4", edgecolor="white", linewidth=0.5, align="center")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "Ø", "S", "V"])
    ax.set_title(title, y=1.15, fontsize=11)


results = {}
for city, coords in cities.items():
    log.info("%s: henter kjørbart gatenett (radius %d m)", city, RADIUS)
    try:
        G = fetch_graph(city, coords)
    except Exception:
        log.exception("%s: kunne ikke hente gatenett, hopper over", city)
        continue
    log.info("%s: %d noder, %d kanter", city, len(G.nodes), len(G.edges))

    # Konverter til urettet graf og beregn kompassretning (bearings)
    Gu = ox.convert.to_undirected(G)
    Gu = ox.bearing.add_edge_bearings(Gu)

    # Orienteringsentropi og normalisert orienteringsorden (Phi, 0–1)
    entropy = ox.bearing.orientation_entropy(Gu, num_bins=NUM_BINS)
    phi = 1 - ((entropy - H_GRID) / (H_MAX - H_GRID)) ** 2
    counts, _ = bearing_histogram(Gu)
    results[city] = (entropy, phi, counts)

    log.info("%-15s | Entropi (H): %.3f | Orden (Phi): %.3f", city, entropy, phi)

    # Ett polardiagram per by
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    polar_plot(ax, counts, f"{city}\nH = {entropy:.3f}, Φ = {phi:.3f}")
    path = OUT_DIR / f"{city.lower()}_polar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("%s: lagret %s", city, path)

if not results:
    raise SystemExit("Ingen byer ble hentet, ingenting å visualisere.")

# Samlet panel med alle byer, sortert etter orden (høyest Phi først)
ranked = sorted(results.items(), key=lambda kv: kv[1][1], reverse=True)
ncols = 3
nrows = math.ceil(len(ranked) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.8 * nrows),
                         subplot_kw={"projection": "polar"})
axes = np.atleast_1d(axes).ravel()
for ax, (city, (entropy, phi, counts)) in zip(axes, ranked):
    polar_plot(ax, counts, f"{city}\nH = {entropy:.3f}, Φ = {phi:.3f}")
for ax in axes[len(ranked):]:
    ax.set_visible(False)
fig.suptitle(f"Orientering av gatenett ({RADIUS / 1000:g} km radius, sortert etter Φ)",
             fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3)
panel = OUT_DIR / "alle_byer_polar.png"
fig.savefig(panel, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Lagret samlet panel: %s", panel)

# Søylediagram: orden (Phi) per by
fig, ax = plt.subplots(figsize=(8, 4.5))
names = [c for c, _ in ranked]
phis = [v[1] for _, v in ranked]
bars = ax.bar(names, phis, color="#1f77b4")
ax.bar_label(bars, fmt="%.3f", padding=3)
ax.set_ylabel("Orienteringsorden (Φ)")
ax.set_ylim(0, 1.1)
ax.set_title("Hvor ordnet er gatenettet? (1 = perfekt rutenett)")
fig.tight_layout()
bar_path = OUT_DIR / "phi_sammenligning.png"
fig.savefig(bar_path, dpi=150)
plt.close(fig)
log.info("Lagret sammenligning: %s", bar_path)

log.info("Ferdig. Figurer ligger i %s", OUT_DIR)
