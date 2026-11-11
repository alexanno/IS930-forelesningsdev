INSTALL spatial; LOAD spatial;

CREATE OR REPLACE TEMP VIEW rest AS
SELECT id, navn, kategori,
       ST_Transform(geometry, 'EPSG:4326', 'EPSG:25833', always_xy := true) AS geom
FROM 'restauranter.parquet';
CREATE OR REPLACE TEMP VIEW brann AS SELECT * FROM ST_Read('brannstasjon.gpkg');
