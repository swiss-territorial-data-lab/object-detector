import requests
import os

from osgeo import gdal
from rasterio.transform import from_bounds

# settings

ROOT_URL = "https://sitn.ne.ch/mapproxy95/service"
BBOX = "763453.0385123404,5969120.412845984,763605.9125689107,5969273.286902554"
WIDTH=256
HEIGHT=256
LAYERS = "ortho2019"
SRS="EPSG:900913"
OUTPUT_DIR = 'output-NE'
# let's make the output directory in case it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PNG_FILENAME = os.path.join(OUTPUT_DIR, 'test.png')

# image download

params = dict(
    service="WMS",
    version="1.1.1",
    request="GetMap",
    layers=LAYERS,
    format="image/png",
    srs=SRS,
    transparent=True,
    styles="",
    bbox=BBOX,
    width=WIDTH,
    height=HEIGHT,
    #exceptions="application/vnd.ogc.se_inimage",
)

r = requests.get(ROOT_URL, params=params)
with open(f'{PNG_FILENAME}', 'wb') as fp:
    fp.write(r.content)

# PGW file generation

xmin, ymin, xmax, ymax = [float(x) for x in BBOX.split(',')]

affine = from_bounds(xmin, ymin, xmax, ymax, WIDTH, HEIGHT)

a = affine.a
b = affine.b
c = affine.c
d = affine.d
e = affine.e
f = affine.f

c += a/2.0 # <- IMPORTANT
f += e/2.0 # <- IMPORTANT

pgw = "\n".join([str(a), str(d), str(b), str(e), str(c), str(f)+"\n"])
print(pgw)

with open(PNG_FILENAME.replace('.png', '.pgw'), 'w') as fp:
    fp.write(pgw)

# PNG + PGW => GeoTIFF

src_ds = gdal.Open(PNG_FILENAME)
gdal.Translate(PNG_FILENAME.replace('.png', '.tiff'), src_ds, options=f'-of GTiff -a_srs {SRS}')
src_ds = None 

print('Output GeoTIFF:', PNG_FILENAME.replace('.png', '.tiff'))