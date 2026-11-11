"""Bygger eksempeldata.qgz og restauranttetthet.gpkg med PyQGIS.

Kjør (QGIS 3.44 på macOS):
  R=/Applications/QGIS.app/Contents/Resources/python3.11
  QT_QPA_PLATFORM=offscreen PYTHONHOME=$R/.. PYTHONPATH=$R:$R/site-packages:$R/lib-dynload \
    /Applications/QGIS.app/Contents/MacOS/python3.12 lag_qgis_prosjekt.py
"""
import os
from qgis.PyQt.QtCore import Qt, QVariant
from qgis.PyQt.QtGui import QColor, QFont
from qgis.core import (
    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsFeature, QgsField, QgsFields, QgsSpatialIndex, QgsVectorFileWriter,
    QgsWkbTypes, QgsSymbol, QgsFillSymbol, QgsLineSymbol, QgsMarkerSymbol, QgsSimpleFillSymbolLayer,
    QgsSingleSymbolRenderer, QgsCategorizedSymbolRenderer, QgsRendererCategory,
    QgsGraduatedSymbolRenderer, QgsRendererRange, QgsHeatmapRenderer, QgsProperty,
    QgsPalLayerSettings, QgsTextFormat, QgsTextBufferSettings, QgsVectorLayerSimpleLabeling,
    QgsRuleBasedLabeling, QgsSymbolLayer, QgsUnitTypes, QgsReferencedRectangle, QgsMapLayer,
    QgsGradientColorRamp, QgsGradientStop, Qgis, QgsBookmark, QgsLayerTreeLayer, QgsGeometry,
)

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
QgsApplication.setPrefixPath("/Applications/QGIS.app/Contents/MacOS", True)
qgs = QgsApplication([], False)
qgs.initQgis()

crs = QgsCoordinateReferenceSystem("EPSG:25833")
wgs = QgsCoordinateReferenceSystem("EPSG:4326")

# ---------------------------------------------------------------- 1. avledet lag: tetthet per eiendom
eiendom = QgsVectorLayer("eiendom.gpkg|layername=eiendom", "eiendom", "ogr")
rest = QgsVectorLayer("restauranter.parquet", "restauranter", "ogr")
assert eiendom.isValid() and rest.isValid(), "kunne ikke lese kildedata"

to_utm = QgsCoordinateTransform(rest.crs(), crs, QgsProject.instance())
idx = QgsSpatialIndex()
pts = {}
for f in rest.getFeatures():
    g = QgsGeometry(f.geometry())
    g.transform(to_utm)
    nf = QgsFeature(f.id())
    nf.setGeometry(g)
    idx.addFeature(nf)
    pts[f.id()] = g

fields = QgsFields()
fields.append(QgsField("gnr_bnr", QVariant.String))
fields.append(QgsField("antall_restauranter", QVariant.Int))
fields.append(QgsField("areal_km2", QVariant.Double))
fields.append(QgsField("restauranter_per_km2", QVariant.Double))

out = "restauranttetthet.gpkg"
if os.path.exists(out):
    os.remove(out)
opts = QgsVectorFileWriter.SaveVectorOptions()
opts.driverName = "GPKG"
opts.layerName = "eiendom_restauranttetthet"
w = QgsVectorFileWriter.create(out, fields, QgsWkbTypes.Polygon, crs,
                               QgsProject.instance().transformContext(), opts)
total = 0
for f in eiendom.getFeatures():
    g = f.geometry()
    n = sum(1 for i in idx.intersects(g.boundingBox()) if g.intersects(pts[i]))
    km2 = g.area() / 1e6
    nf = QgsFeature(fields)
    nf.setGeometry(g)
    nf.setAttributes([f["gnr_bnr"], n, round(km2, 4), round(n / km2, 2)])
    w.addFeature(nf)
    total += n
del w
print(f"{total} av {rest.featureCount()} restauranter ligger innenfor eiendommene")

# ---------------------------------------------------------------- 2. lag i prosjektet
def vl(path, name, layer=None):
    src = f"{path}|layername={layer}" if layer else path
    l = QgsVectorLayer(src, name, "ogr")
    assert l.isValid(), path
    return l

kommuner = vl("kommuner.gpkg", "Kommuner", "kommuner")
grunnkrets = vl("grunnkrets.gpkg", "Grunnkretser", "grunnkrets")
faresone = vl("faresone.gpkg", "Faresone (flom)", "faresone")
buffer_ = vl("buffer_duckdb.gpkg", "Buffer (DuckDB)", "buffer_duckdb")
vassdrag = vl("vassdrag.gpkg", "Vassdrag", "vassdrag")
bygg = vl("bygg.gpkg", "Bygg", "bygg")
brann = vl("brannstasjon.gpkg", "Brannstasjoner", "brannstasjon")
tetthet = vl("restauranttetthet.gpkg", "Restauranttetthet per eiendom", "eiendom_restauranttetthet")
restl = vl("restauranter.parquet", "Restauranter", None)
heat = vl("restauranter.parquet", "Restauranter – heatmap", None)
eiendom_l = vl("eiendom.gpkg", "Eiendommer", "eiendom")
base = QgsRasterLayer(
    "type=xyz&url=https://basemaps.cartocdn.com/light_all/%7Bz%7D/%7Bx%7D/%7By%7D.png&zmax=19&zmin=0",
    "Bakgrunnskart (CARTO Positron © OSM)", "wms")


def c(h, a=255):
    col = QColor(h)
    col.setAlpha(a)
    return col


def fill(layer, color, outline, w=0.26, alpha=255, style="solid", dash=None):
    s = QgsFillSymbol.createSimple({
        "color": c(color, alpha).name(QColor.HexArgb), "outline_color": outline,
        "outline_width": str(w), "style": style,
        "outline_style": dash or "solid"})
    layer.setRenderer(QgsSingleSymbolRenderer(s))


def label(layer, expr, size=8, color="#333333", buf="#ffffff", is_expr=False, placement=None, bold=False):
    s = QgsPalLayerSettings()
    s.fieldName = expr
    s.isExpression = is_expr
    tf = QgsTextFormat()
    f = QFont("Helvetica Neue", int(size))
    f.setBold(bold)
    tf.setFont(f)
    tf.setSize(size)
    tf.setColor(QColor(color))
    b = QgsTextBufferSettings()
    b.setEnabled(True)
    b.setSize(1.0)
    b.setColor(QColor(buf))
    tf.setBuffer(b)
    s.setFormat(tf)
    if placement is not None:
        s.placement = placement
    layer.setLabelsEnabled(True)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(s))


# Kommuner: lyse pastellflater med tydelig, stiplet grense + kommunenavn
cat = QgsCategorizedSymbolRenderer("kommunenavn", [
    QgsRendererCategory("Kristiansand", QgsFillSymbol.createSimple(
        {"color": "#e9f0e1", "outline_color": "#5b6b4f", "outline_width": "0.8", "outline_style": "dash"}), "Kristiansand"),
    QgsRendererCategory("Vennesla", QgsFillSymbol.createSimple(
        {"color": "#f1e8d6", "outline_color": "#7a6a45", "outline_width": "0.8", "outline_style": "dash"}), "Vennesla"),
])
kommuner.setRenderer(cat)
kommuner.setOpacity(0.55)
label(kommuner, "upper(kommunenavn)", size=14, color="#4a5a40", is_expr=True, bold=True)

# Grunnkretser: tynne grå linjer + nummer
fill(grunnkrets, "#ffffff", "#9aa0a6", w=0.2, alpha=0)
label(grunnkrets, "right(grunnkretsnummer, 2)", size=7, color="#8a9097", is_expr=True)

# Faresone: gjennomsiktig blå med skravur
fz = QgsFillSymbol.createSimple({"color": "#3b82f6", "outline_color": "#1d4ed8", "outline_width": "0.4", "style": "solid"})
fz.setOpacity(0.28)
faresone.setRenderer(QgsSingleSymbolRenderer(fz))

# Buffer fra DuckDB: rød stiplet kontur uten fyll
fill(buffer_, "#ef4444", "#b91c1c", w=0.6, alpha=0, dash="dash")

# Eiendommer (kildedata): lys kontur (brukes som referanse, skjult under tetthetslaget)
fill(eiendom_l, "#ffffff", "#7c7c7c", w=0.15, alpha=0)

# Vassdrag: blå linjer, Otra tykkere enn bekk
vcat = QgsCategorizedSymbolRenderer("navn", [
    QgsRendererCategory("Otra", QgsLineSymbol.createSimple({"color": "#1e6fd9", "width": "1.6", "capstyle": "round"}), "Otra"),
    QgsRendererCategory("Bekk", QgsLineSymbol.createSimple({"color": "#5aa7f0", "width": "0.7", "capstyle": "round"}), "Bekk"),
])
vassdrag.setRenderer(vcat)
label(vassdrag, "navn", size=9, color="#1e4fa0")
ls = vassdrag.labeling().settings()
ls.placement = Qgis.LabelPlacement.Line
vassdrag.setLabeling(QgsVectorLayerSimpleLabeling(ls))

# Bygg: små prikker, størrelse etter bruksareal
bs = QgsMarkerSymbol.createSimple({"name": "circle", "color": "#6b7280", "outline_color": "#ffffff", "outline_width": "0.1", "size": "1.4"})
bs.setOpacity(0.8)
bs.symbolLayer(0).setDataDefinedProperty(
    QgsSymbolLayer.PropertySize,
    QgsProperty.fromExpression('scale_linear("bruksareal", 30, 400, 1.2, 3.2)'))
bygg.setRenderer(QgsSingleSymbolRenderer(bs))

# Brannstasjoner: rødt kvadrat med hvit kant + navn
br = QgsMarkerSymbol.createSimple({"name": "square", "color": "#dc2626", "outline_color": "#ffffff", "outline_width": "0.6", "size": "4.2"})
brann.setRenderer(QgsSingleSymbolRenderer(br))
label(brann, "navn", size=9, color="#991b1b", bold=True)
ls = brann.labeling().settings()
ls.placement = Qgis.LabelPlacement.AroundPoint
ls.dist = 2
brann.setLabeling(QgsVectorLayerSimpleLabeling(ls))

# Restauranter: oransje prikker (tydelig mot grå bygg)
rs = QgsMarkerSymbol.createSimple({"name": "circle", "color": "#f97316", "outline_color": "#7c2d12", "outline_width": "0.3", "size": "2.6"})
restl.setRenderer(QgsSingleSymbolRenderer(rs))
restl.setDisplayExpression('"navn"')

# Heatmap av restauranter (skrudd av som standard)
hm = QgsHeatmapRenderer()
hm.setRadius(12)
hm.setRadiusUnit(QgsUnitTypes.RenderMillimeters)
ramp = QgsGradientColorRamp(c("#fff7ec", 0), c("#7f0000"), False,
                            [QgsGradientStop(0.25, c("#fdd49e")), QgsGradientStop(0.5, c("#fc8d59")),
                             QgsGradientStop(0.75, c("#d7301f"))])
hm.setColorRamp(ramp)
hm.setMaximumValue(0)
hm.setRenderQuality(3)
heat.setRenderer(hm)

# Restauranttetthet per eiendom: gradert, ColorBrewer YlOrRd
fmax = max(f["restauranter_per_km2"] for f in tetthet.getFeatures())
nmax = max(f["antall_restauranter"] for f in tetthet.getFeatures())
print("maks antall per eiendom:", nmax, "– maks tetthet:", fmax, "per km²")
step = 1 / 0.16  # tetthet for én restaurant i en 400x400 m-eiendom
def klasse(lo, hi):  # grenser i antall restauranter -> tetthet per km²
    return (lo - 0.5) * step, (hi + 0.5) * step


classes = [
    ("0 (ingen)", -0.001, 0.001, "#f7f7f2"),
    ("1 restaurant", *klasse(1, 1), "#fed976"),
    ("2–3 restauranter", *klasse(2, 3), "#feb24c"),
    ("4–15 restauranter", *klasse(4, 15), "#fd8d3c"),
    ("16–30 restauranter", *klasse(16, 30), "#e31a1c"),
    ("Over 30 restauranter", *klasse(31, 10**6), "#800026"),
]
ranges = []
for lbl, lo, hi, col in classes:
    sym = QgsFillSymbol.createSimple({"color": col, "outline_color": "#ffffff", "outline_width": "0.2"})
    ranges.append(QgsRendererRange(lo, hi, sym, lbl))
grad = QgsGraduatedSymbolRenderer("restauranter_per_km2", ranges)
tetthet.setRenderer(grad)
tetthet.setOpacity(0.85)
# vis antall som tall på eiendommer som har restauranter
lab = QgsPalLayerSettings()
lab.fieldName = 'if("antall_restauranter" > 0, to_string("antall_restauranter"), \'\')'
lab.isExpression = True
tf = QgsTextFormat()
fnt = QFont("Helvetica Neue", 9)
fnt.setBold(True)
tf.setFont(fnt)
tf.setSize(9)
tf.setColor(QColor("#3b1f0b"))

bb = QgsTextBufferSettings()
bb.setEnabled(True)
bb.setSize(0.9)
bb.setColor(QColor("#ffffff"))
tf.setBuffer(bb)
lab.setFormat(tf)
lab.placement = Qgis.LabelPlacement.OverPoint
tetthet.setLabeling(QgsVectorLayerSimpleLabeling(lab))
tetthet.setLabelsEnabled(True)
tetthet.setMapTipTemplate(
    "<b>Eiendom [% \"gnr_bnr\" %]</b><br>[% \"antall_restauranter\" %] restauranter<br>"
    "[% round(\"restauranter_per_km2\", 1) %] per km²")

# ---------------------------------------------------------------- 3. lagtre (rekkefølge: øverst først)
proj = QgsProject.instance()
proj.clear()
proj.setTitle("Eksempeldata – Kristiansand/Vennesla")
proj.setCrs(crs)
proj.setFilePathStorage(Qgis.FilePathType.Relative)

for l in (kommuner, grunnkrets, faresone, buffer_, eiendom_l, tetthet, vassdrag, bygg, restl, heat, brann, base):
    proj.addMapLayer(l, False)
root = proj.layerTreeRoot()


def group(name, layers, checked=True, expanded=True):
    g = root.addGroup(name)
    for l, vis in layers:
        n = g.addLayer(l)
        n.setItemVisibilityChecked(vis)
    g.setExpanded(expanded)
    return g


group("Punkt og linje", [(brann, True), (restl, True), (bygg, True), (vassdrag, True)])
group("Analyse: restauranttetthet", [(heat, False), (tetthet, True)])
group("Flater", [(buffer_, True), (faresone, True), (eiendom_l, False), (grunnkrets, True), (kommuner, True)])
group("Bakgrunn", [(base, True)], expanded=False)

# flytt tetthet til toppen av flatene (rett under linjer/punkter) – rekkefølgen over gir dette allerede
proj.setBackgroundColor(QColor("#f4f6f8"))

# kartutsnitt: hele området, litt luft
ext = grunnkrets.extent()
ext.scale(1.05)
proj.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(ext, crs))
proj.viewSettings().setPresetFullExtent(QgsReferencedRectangle(grunnkrets.extent(), crs))

# bokmerke for eiendomsområdet med restauranttetthet
bm = QgsBookmark()
bm.setName("Eiendommer – restauranttetthet")
bm.setExtent(QgsReferencedRectangle(tetthet.extent(), crs))
proj.bookmarkManager().addBookmark(bm)
bm2 = QgsBookmark()
bm2.setName("Hele området")
bm2.setExtent(QgsReferencedRectangle(ext, crs))
proj.bookmarkManager().addBookmark(bm2)

ok = proj.write(os.path.join(here, "eksempeldata.qgz"))
print("lagret eksempeldata.qgz:", ok)
qgs.exitQgis()
