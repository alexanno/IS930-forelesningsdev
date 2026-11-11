INSTALL spatial; LOAD spatial;

CREATE OR REPLACE VIEW bygg AS SELECT * FROM ST_Read('bygg.gpkg');
CREATE OR REPLACE VIEW brann AS SELECT * FROM ST_Read('brannstasjon.gpkg');
CREATE OR REPLACE VIEW eiendom AS SELECT * FROM ST_Read('eiendom.gpkg');
CREATE OR REPLACE VIEW kommune AS SELECT * FROM ST_Read('kommuner.gpkg');

.print '--- 1. Bygninger innen 100 m fra brannstasjon'
SELECT b.bygg_id, s.navn AS brannstasjon, round(ST_Distance(b.geom, s.geom), 1) AS avstand_m
FROM bygg b JOIN brann s ON ST_DWithin(b.geom, s.geom, 100)
ORDER BY avstand_m;

.print '--- 2. Brannstasjoner i eiendom'
SELECT s.navn, e.gnr_bnr
FROM brann s JOIN eiendom e ON ST_Intersects(e.geom, s.geom);

.print '--- 3. Eiendommer per kommune'
SELECT k.kommunenavn, count(e.gnr_bnr) AS antall_eiendommer
FROM kommune k LEFT JOIN eiendom e ON ST_Intersects(k.geom, ST_PointOnSurface(e.geom))
GROUP BY k.kommunenavn ORDER BY 1;

.print '--- 4a. Bygninger per kommune'
SELECT k.kommunenavn, count(b.bygg_id) AS antall_bygg
FROM kommune k LEFT JOIN bygg b ON ST_Intersects(k.geom, b.geom)
GROUP BY k.kommunenavn ORDER BY 1;

.print '--- 4b. Bygninger per eiendom (kun eiendommer med bygg)'
SELECT e.gnr_bnr, count(*) AS antall_bygg
FROM eiendom e JOIN bygg b ON ST_Intersects(e.geom, b.geom)
GROUP BY e.gnr_bnr ORDER BY antall_bygg DESC, e.gnr_bnr;
