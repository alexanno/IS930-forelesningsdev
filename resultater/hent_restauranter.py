"""Henter restauranter fra siste Overture Maps-release (via STAC) innenfor kommune-bbox til GeoParquet."""
import json, urllib.request
import duckdb

STAC = "https://stac.overturemaps.org"
get = lambda u: json.load(urllib.request.urlopen(u))

con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")

# 1. bbox i WGS84 fra kommunelaget (EPSG:25833)
xmin, ymin, xmax, ymax = con.sql("""
    WITH b AS (SELECT ST_Extent(ST_Transform(geom, 'EPSG:25833', 'EPSG:4326', always_xy := true)) AS e
               FROM ST_Read('kommuner.gpkg'))
    SELECT ST_XMin(e), ST_YMin(e), ST_XMax(e), ST_YMax(e) FROM b""").fetchone()
print("bbox:", xmin, ymin, xmax, ymax)

# 2. siste release fra STAC, og kun filene (items) som overlapper bbox
release = get(f"{STAC}/catalog.json")["latest"]
print("release:", release)
coll = get(f"{STAC}/{release}/places/place/collection.json")
files = []
for l in coll["links"]:
    if l["rel"] != "item":
        continue
    item = get(l["href"])
    bx0, by0, bx1, by1 = item["bbox"]
    if bx0 <= xmax and bx1 >= xmin and by0 <= ymax and by1 >= ymin:
        files.append(item["assets"]["aws"]["alternate"]["s3"]["href"])
print("filer som overlapper bbox:", len(files))

# 3. Les kun bbox + kategori (bbox-struct gir row group-pruning), skriv GeoParquet
con.execute(f"""
COPY (
  SELECT id,
         names.primary AS navn,
         taxonomy.primary AS kategori,
         basic_category,
         confidence,
         addresses[1].freeform AS adresse,
         addresses[1].locality AS sted,
         geometry
  FROM read_parquet({files}, hive_partitioning=1)
  WHERE bbox.xmin >= {xmin} AND bbox.xmax <= {xmax}
    AND bbox.ymin >= {ymin} AND bbox.ymax <= {ymax}
    AND basic_category = 'restaurant'
) TO 'restauranter.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
""")
print(con.sql("SELECT count(*) AS antall FROM 'restauranter.parquet'"))
