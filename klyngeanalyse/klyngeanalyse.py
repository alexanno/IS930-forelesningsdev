# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Nærhet, tetthet og klynger: kNN og DBSCAN
#
# Tre spørsmål vi stiller til punktdata, med serveringssteder (restaurant, kafé, bar, pub, fast food)
# fra OpenStreetMap i Kristiansand sentrum som eksempel:
#
# 1. **Nærhet**: hvem er nærmest hvem? (*k nearest neighbours*, kNN)
# 2. **Tetthet**: hvor er det mest av noe? (rutenett og kernel density)
# 3. **Klynger**: hvor ligger ting i grupper? (**DBSCAN**, og k-means som kontrast)
#
# Hver analyse vises i Python (geopandas / scikit-learn), og som SQL for **PostGIS**.
#
# | | Python | PostGIS |
# |---|---|---|
# | kNN | `NearestNeighbors`, `sjoin_nearest` | `<->` + `LATERAL` |
# | DBSCAN | `sklearn.cluster.DBSCAN` | `ST_ClusterDBSCAN` |
# | k-means | `sklearn.cluster.KMeans` | `ST_ClusterKMeans` |
#
# **PostGIS er ikke installert her**, så PostGIS-eksemplene står som kodeblokker som ikke er kjørt.

# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KernelDensity, NearestNeighbors

ox.settings.use_cache = True
ox.settings.cache_folder = "cache"
ox.settings.log_console = False

DATA = Path("data")
DATA.mkdir(exist_ok=True)

# Okabe–Ito (fargeblind-sikker) til klynger, grått til støy
FARGER = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
STOY = "#B0B0B0"

# %% [markdown]
# ## Data: serveringssteder i Kristiansand sentrum
#
# Hentes én gang fra OSM (3 km rundt stasjonen) og lagres i `data/servering.gpkg`.
# **Viktig:** vi projiserer til UTM 33N (EPSG:25833), slik at avstander er i **meter**.
# Både `eps` i DBSCAN og `ST_DWithin` i SQL forutsetter at koordinatene har meter som enhet.
# Med lengde/bredde (grader) ville `eps = 150` bety 150 grader.

# %%
SERVERING_FIL = DATA / "servering.gpkg"
if SERVERING_FIL.exists():
    servering = gpd.read_file(SERVERING_FIL)
else:
    servering = ox.features.features_from_point(
        (58.1467, 7.9956),  # Kristiansand stasjon
        tags={"amenity": ["restaurant", "cafe", "bar", "pub", "fast_food"]},
        dist=3000,
    )
    servering = servering.assign(geometry=servering.representative_point())[["name", "amenity", "geometry"]]
    servering = servering.reset_index(drop=True)
    servering.to_file(SERVERING_FIL)

servering = servering.to_crs(25833)
servering["id"] = range(len(servering))
servering["x"] = servering.geometry.x
servering["y"] = servering.geometry.y
XY = servering[["x", "y"]].to_numpy()

print(len(servering), "serveringssteder")
servering["amenity"].value_counts().to_frame()

# %%
fig, ax = plt.subplots(figsize=(7, 6))
servering.plot(ax=ax, markersize=18, color=FARGER[0])
ax.set_title("Serveringssteder i Kristiansand sentrum (OSM)")
ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## 1. Nærhet: k nærmeste naboer (kNN)
#
# kNN-søk: for hvert punkt, finn de *k* nærmeste punktene. Avstanden her er **luftlinje**.
# Langs vegnett eller i reisetid kan svaret bli et helt annet (se `../osmnx-demo/kristiansand_osmnx.ipynb`).
#
# *Merk:* «kNN» brukes også om en klassifikasjonsmetode (*k-nearest neighbours classifier*:
# et punkt får klassen til flertallet av de k nærmeste). Her bruker vi kNN som **søk** etter naboer.
# Det er også byggesteinen i DBSCAN, som vi kommer til under.

# %% [markdown]
# ### Python: scikit-learn og geopandas
#
# Vi finner de tre nærmeste naboene til hvert sted. Første nabo er punktet selv (avstand 0), så vi ber om k+1.

# %%
K = 3
nn = NearestNeighbors(n_neighbors=K + 1).fit(XY)
avstand, indeks = nn.kneighbors(XY)

servering["avstand_nabo1"] = avstand[:, 1]
servering["avstand_nabo3"] = avstand[:, K]
servering[["name", "amenity", "avstand_nabo1", "avstand_nabo3"]].sort_values("avstand_nabo3").head(5)

# %% [markdown]
# Til det nærmeste stedet av en *annen type* bruker vi `sjoin_nearest` i geopandas.
# Eksempel: hvor langt har hver restaurant til nærmeste pub eller bar?

# %%
puber = servering[servering["amenity"].isin(["pub", "bar"])][["name", "geometry"]]
restauranter = servering[servering["amenity"] == "restaurant"][["name", "geometry"]]

til_pub = restauranter.sjoin_nearest(puber, how="left", distance_col="avstand_m", rsuffix="pub")
til_pub[["name_left", "name_pub", "avstand_m"]].sort_values("avstand_m").head(5).round(0)

# %% [markdown]
# ### PostGIS (ikke kjørt her)
#
# Lastes inn med f.eks. `ogr2ogr -f PostgreSQL PG:"dbname=gis" data/servering.gpkg -nln servering -t_srs EPSG:25833`.
# Operatoren `<->` gir avstand og kan bruke en GiST-indeks. Den gjør kNN-søk raskt, fordi PostGIS
# slipper å regne ut avstand til alle punkter.
#
# ```sql
# CREATE INDEX servering_geom_idx ON servering USING GIST (geom);
#
# -- de 3 nærmeste naboene til hvert sted
# SELECT a.name AS fra, n.name AS til, n.avstand_m
# FROM servering a
# CROSS JOIN LATERAL (
#     SELECT b.name, a.geom <-> b.geom AS avstand_m
#     FROM servering b
#     WHERE b.id <> a.id
#     ORDER BY a.geom <-> b.geom
#     LIMIT 3
# ) n;
#
# -- alt innenfor en terskel (her 150 m), bruker også indeksen
# SELECT a.name, count(*) AS antall_naboer
# FROM servering a
# JOIN servering b ON ST_DWithin(a.geom, b.geom, 150) AND a.id <> b.id
# GROUP BY a.name;
# ```

# %% [markdown]
# ## 2. Tetthet: hvor er det mest?
#
# Før vi finner klynger: den enkleste måten å se «mest av noe» på er å **telle per rute**.
# Her ruter på 250 m (som SSB-rutenettet). Resultatet avhenger av **cellestørrelsen** og av
# hvor rutenettet starter. Samme data gir ulike kart. Kernel density (KDE) glatter ut
# rutegrensene, men da må du velge **bandwidth** i stedet.

# %%
def tell_per_rute(xy, celle):
    ix = np.floor(xy[:, 0] / celle).astype(int)
    iy = np.floor(xy[:, 1] / celle).astype(int)
    return pd.Series(1, index=pd.MultiIndex.from_arrays([ix, iy])).groupby(level=[0, 1]).sum()


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
xmin, ymin = XY.min(axis=0) - 200
xmax, ymax = XY.max(axis=0) + 200

for ax, celle in zip(axes[:2], [250, 500]):
    ax.hist2d(XY[:, 0], XY[:, 1], bins=[np.arange(xmin, xmax, celle), np.arange(ymin, ymax, celle)],
              cmap="viridis", cmin=1)
    ax.set_title(f"Antall per {celle} m-rute")

BANDBREDDE = 150
kde = KernelDensity(bandwidth=BANDBREDDE).fit(XY)
gx, gy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
tetthet = np.exp(kde.score_samples(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
axes[2].contourf(gx, gy, tetthet * 1e6, levels=12, cmap="viridis")
axes[2].set_title(f"KDE, bandwidth {BANDBREDDE} m")

for ax in axes:
    ax.scatter(XY[:, 0], XY[:, 1], s=5, c="white", edgecolor="k", linewidth=0.3)
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# Tetthet gir en **kontinuerlig flate**, men ingen avgrensede grupper. Spørsmålet «hvor mange
# klynger er det, og hvilke steder hører til hvilken?» krever klyngeanalyse.
#
# ## 3. Klynger med DBSCAN
#
# DBSCAN (*Density-Based Spatial Clustering of Applications with Noise*) har to parametre:
#
# - **`eps`**: maks avstand mellom to punkter for at de regnes som naboer
# - **`min_samples`**: minste antall punkter (inkl. punktet selv) innen `eps` for at et punkt er et **kjernepunkt**
#
# Algoritmen:
#
# 1. **Kjernepunkt**: har minst `min_samples` punkter innen `eps`.
# 2. **Klynge**: kjernepunkter som ligger innen `eps` av hverandre, henger sammen i samme klynge (kjeder av kjernepunkter gir klynger i vilkårlig form).
# 3. **Randpunkt**: ikke kjerne selv, men innen `eps` av et kjernepunkt. Tilhører den klyngen.
# 4. **Støy**: alle andre. Får etikett `-1` (i PostGIS `NULL`).
#
# Du trenger ikke velge antall klynger på forhånd.

# %% [markdown]
# ### Velge `eps`: k-distance-plott
#
# Tommelfingerregel: sett `min_samples` ut fra hva som er «en gruppe» i problemet ditt, og se på
# avstanden til nabo nr. `min_samples` for alle punkter, sortert. `eps` velges der kurven knekker:
# til venstre ligger punkter i tette områder, til høyre spredte punkter (støy).
# Knekken er ofte tvetydig. Se derfor alltid på kartet og forklar valget.

# %%
MIN_SAMPLES = 5
d_k = np.sort(NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(XY).kneighbors(XY)[0][:, -1])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(d_k, color=FARGER[0])
ax.axhline(150, color=FARGER[1], linestyle="--", label="eps = 150 m")
ax.set_xlabel("Punkter sortert etter avstand")
ax.set_ylabel(f"Avstand til nabo nr. {MIN_SAMPLES} (m)")
ax.set_title("k-distance-plott")
ax.legend()
plt.show()

# %%
def kartlegg(ax, xy, etikett, tittel):
    ax.scatter(*xy[etikett == -1].T, s=14, c=STOY, label="Støy")
    for i, k in enumerate(sorted(set(etikett) - {-1})):
        ax.scatter(*xy[etikett == k].T, s=24, c=FARGER[i % len(FARGER)], edgecolor="k", linewidth=0.3)
    ant = len(set(etikett) - {-1})
    ax.set_title(f"{tittel}\n{ant} klynger, {(etikett == -1).sum()} støy")
    ax.set_aspect("equal")


EPS = 150
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(XY)
servering["dbscan"] = db.labels_

fig, ax = plt.subplots(figsize=(7, 6))
kartlegg(ax, XY, db.labels_, f"DBSCAN eps={EPS} m, min_samples={MIN_SAMPLES}")
ax.legend(loc="lower left")
plt.show()

servering.groupby("dbscan").agg(antall=("id", "size"), kafeer=("amenity", lambda s: (s == "cafe").sum()))

# %% [markdown]
# ### Parametrene styrer svaret
#
# Samme data, tre innstillinger. Små `eps` gir mange små klynger og mye støy. Store `eps` slår
# alt sammen til én stor klynge. **Parametrene må følge med kartet.**

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, (eps, ms) in zip(axes, [(75, 4), (150, 5), (300, 5)]):
    kartlegg(ax, XY, DBSCAN(eps=eps, min_samples=ms).fit_predict(XY), f"eps={eps} m, min_samples={ms}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### K-means som kontrast
#
# K-means krever at du velger antall klynger (*k*) og plasserer **alle** punkter i en klynge. Det finnes
# ingen støy, og klyngene blir runde og jevnt store. Det passer for «del opp i k distrikter», mens DBSCAN
# passer for «finn de tette områdene». (Ikke bland sammen k i k-means og k i kNN.)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
kartlegg(axes[0], XY, KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(XY), "k-means, k=3")
kartlegg(axes[1], XY, db.labels_, f"DBSCAN eps={EPS} m, min_samples={MIN_SAMPLES}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### PostGIS (ikke kjørt her)
#
# PostGIS har klyngefunksjoner som **vindusfunksjoner**. Kolonnen må være i et projisert koordinatsystem
# (meter). Støy får `NULL`.
#
# ```sql
# -- DBSCAN: eps i meter, minpoints tilsvarer min_samples
# CREATE TABLE servering_klynge AS
# SELECT id, name, geom,
#        ST_ClusterDBSCAN(geom, eps := 150, minpoints := 5) OVER () AS klynge
# FROM servering;
#
# -- antall og utstrekning per klynge (støy = NULL hoppes over)
# SELECT klynge,
#        count(*) AS antall,
#        ST_ConvexHull(ST_Collect(geom)) AS omriss
# FROM servering_klynge
# WHERE klynge IS NOT NULL
# GROUP BY klynge;
#
# -- k-means med k = 3
# SELECT id, ST_ClusterKMeans(geom, 3) OVER () AS klynge FROM servering;
#
# -- enklere variant uten min_samples: alt som henger sammen innen 150 m (single linkage)
# SELECT unnest(ST_ClusterWithin(geom, 150)) FROM servering;
# ```
#
# Sjekk at `minpoints` teller punktet selv på samme måte som `min_samples` i din PostGIS-versjon,
# og sammenlign gjerne med scikit-learn på samme data.

# %% [markdown]
# ### Klyngeomriss
#
# Konveks omhylling rundt hver klynge, tilsvarende `ST_ConvexHull(ST_Collect(...))` i PostGIS:

# %%
servering["klynge"] = np.where(db.labels_ == -1, np.nan, db.labels_)
omriss = (servering.dropna(subset=["klynge"])
          .dissolve(by="klynge", aggfunc={"id": "size"})
          .convex_hull.to_frame("geometry").set_geometry("geometry", crs=servering.crs)
          .reset_index())

fig, ax = plt.subplots(figsize=(7, 6))
omriss.plot(ax=ax, alpha=0.25, color=FARGER[1], edgecolor=FARGER[1])
servering.plot(ax=ax, markersize=14, color=STOY)
servering.dropna(subset=["klynge"]).plot(ax=ax, markersize=22, color=FARGER[0])
ax.set_title("Klyngeomriss (konveks omhylling) fra DBSCAN")
ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## Hva må du være oppmerksom på?
#
# Samme valg som for alle nærhets-, tetthets- og dekningsanalyser:
#
# - **Parametre**: `eps`, `min_samples`, cellestørrelse og bandwidth bestemmer svaret. Rapportér dem sammen med kartet.
# - **Avstandsmål**: her luftlinje. Langs vegnett (osmnx) eller i reisetid kan nærhet se helt annerledes ut, særlig i en fjordkommune.
# - **Normalisering**: klynger av serveringssteder følger befolkning og turisme. Vis antall per areal eller innbygger, ikke bare antall.
# - **Datakildens dekning**: OSM er tettere i by enn i distrikt. Færre punkter utenfor sentrum kan være manglende registrering.
# - **Klynge er ikke det samme som sammenheng**: DBSCAN finner tette områder, ikke årsaker. En klynge er en hypotese du må forklare.
# - **Ulik tetthet**: DBSCAN bruker én `eps` for alle. Har du både tett by og spredt omland, får du enten for store eller for mange klynger (se `HDBSCAN` i scikit-learn).
# - **Koordinatsystem**: bruk projisert CRS i meter. Grader gir feil avstand, og feil avstand gir feil klynger.
#
# ### Oppgaver
#
# 1. Endre `EPS` og `MIN_SAMPLES`. Hvordan endrer antall klynger og støy seg?
# 2. Kjør DBSCAN på bare restauranter, og på bare kaféer. Ligger klyngene på samme sted?
# 3. Bytt til `eksempeldata/bygg.gpkg` (500 jevnt spredte bygg). Finner DBSCAN klynger? Hva forteller det om data uten tydelig gruppering?
# 4. Skriv `ST_ClusterWithin`-varianten i Python (hint: sammenhengende komponenter i en graf der kantene er par innen 150 m). Hvordan skiller den seg fra DBSCAN?
