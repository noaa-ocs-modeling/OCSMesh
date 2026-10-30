from pathlib import Path

# Small public DEM (81 x 81 pixels, New York Harbor, NAD83 / NAVD88) for
# the tests that need real topography. It is NCEI's 1/9 arc-second
# Sandy topobathy tile ncei19_n40x75_w073x75_2015v1.tif from
# https://chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/
# resampled with Resampling.average and scaling_factor=0.01. It is kept
# in the repository so the tests need no network access.
TEST_FILE = str(Path(__file__).resolve().parents[1] / 'data' / 'test_dem.tif')
