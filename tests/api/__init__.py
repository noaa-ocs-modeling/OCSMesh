import os
os.system("""
    ls /tmp/test_dem.tif > /dev/null
    if [ $? -eq 0 ]; then exit; fi
    wget https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/northeast_sandy/ncei19_n40x75_w073x75_2015v1.tif -O /tmp/fullsize_dem.tif
    gdalwarp -tr 0.0005 0.0005 -r average -overwrite /tmp/fullsize_dem.tif /tmp/test_dem.tif
""")
