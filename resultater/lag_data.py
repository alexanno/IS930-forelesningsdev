import geopandas as gpd, numpy as np
from shapely.geometry import Point, LineString, box
from pyproj import Transformer
rng = np.random.default_rng(1)
tr = Transformer.from_crs(4326, 25833, always_xy=True)
x0, y0 = tr.transform(8.0, 58.15)       # Kristiansand
crs = "EPSG:25833"
# kommuner: to ruter side om side
kommuner = gpd.GeoDataFrame({"kommunenavn": ["Kristiansand", "Vennesla"], "kommunenummer": ["4204", "4223"]},
    geometry=[box(x0-6000, y0-6000, x0+4000, y0+6000), box(x0+4000, y0-6000, x0+12000, y0+6000)], crs=crs)
# grunnkretser: 4x3 ruter
gk = []
for ix in range(5):
    for iy in range(3):
        gk.append(box(x0-6000+ix*3600, y0-6000+iy*4000, x0-6000+(ix+1)*3600, y0-6000+(iy+1)*4000))
gk = gpd.GeoDataFrame({"grunnkretsnummer": [f"{42040000+i}" for i in range(len(gk))]}, geometry=gk, crs=crs)
gk["kommunenummer"] = np.where(gk.geometry.centroid.x < x0+4000, "4204", "4223")
# bygg (punkter)
pts = [Point(x0-6000+rng.random()*18000, y0-6000+rng.random()*12000) for _ in range(500)]
bygg = gpd.GeoDataFrame({"bygg_id": range(500), "bruksareal": rng.integers(30, 400, 500)}, geometry=pts, crs=crs)
# vassdrag
vass = gpd.GeoDataFrame({"navn": ["Otra", "Bekk"]}, geometry=[LineString([(x0-5000, y0-5500), (x0, y0), (x0+6000, y0+5500)]), LineString([(x0+3000, y0-5000), (x0+3500, y0+2000)])], crs=crs)
# brannstasjoner
brann = gpd.GeoDataFrame({"navn": ["Brannstasjon A", "Brannstasjon B", "Brannstasjon C"]}, geometry=[Point(x0-2000, y0), Point(x0+5000, y0+3000), Point(x0+9000, y0-3000)], crs=crs)
# eiendommer (rutenett av små polygoner) og faresone
ei = [box(x0-2000+i*400, y0-2000+j*400, x0-2000+(i+1)*400, y0-2000+(j+1)*400) for i in range(10) for j in range(10)]
eiendom = gpd.GeoDataFrame({"gnr_bnr": [f"{i}/1" for i in range(100)]}, geometry=ei, crs=crs)
fare = gpd.GeoDataFrame({"fare": ["flom"]}, geometry=[vass.geometry.iloc[0].buffer(500)], crs=crs)
for n, g in dict(kommuner=kommuner, grunnkrets=gk, bygg=bygg, vassdrag=vass, brannstasjon=brann, eiendom=eiendom, faresone=fare).items():
    g.to_file(f"{n}.gpkg", layer=n, driver="GPKG")
print("ok", {n: len(g) for n, g in dict(kommuner=kommuner, grunnkrets=gk, bygg=bygg).items()})
