import os
import tempfile
import urllib.request
from pathlib import Path

from rasterio.enums import Resampling

from ocsmesh.raster import Raster


# Find a better way!
tif_url = (
    'https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/ncei19_n40x75_w073x75_2015v1.tif'
)
TEST_FILE = os.path.join(tempfile.gettempdir(), 'test_dem.tif')
if not Path(TEST_FILE).exists():
    tmpfd, tmppath = tempfile.mkstemp()
    urllib.request.urlretrieve(tif_url, filename=tmppath)
    os.close(tmpfd)
    r = Raster(tmppath)
    r.resampling_method = Resampling.average
    r.resample(scaling_factor=0.01)
    r.save(TEST_FILE)
