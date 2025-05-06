"""
References
----------
(from more general to more specific)
- https://www.nature.com/articles/s41467-025-56906-7/tables/1
- https://hub.worldpop.org/project/categories?id=3
- https://hub.worldpop.org/geodata/listing?id=64
- https://hub.worldpop.org/geodata/summary?id=24777
"""
import numpy as np
import xarray as xa
import rasterio
from rasterio.transform import xy
from rimeX.datasets.manager import require_dataset

def download_population_count():
    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif"
    local_file_path = require_dataset("worldpop/ppp_2020_1km_Aggregated.tif", url=url)
    return local_file_path

def _get_coords(src):

    l, b, r, t = src.bounds

    lon0, lat0 = xy(src.transform, 0, 0)
    lon1, lat1 = xy(src.transform, src.height-1, src.width-1)
    lon1_check, lat0_check = xy(src.transform, 0, src.width-1)
    lon0_check, lat1_check = xy(src.transform, src.height-1, 0)

    assert lon0 == lon0_check, f"lon0 {lon0} != lon0_check {lon0_check}"
    assert lat0 == lat0_check, f"lat0 {lat0} != lat0_check {lat0_check}"
    assert lon1 == lon1_check, f"lon1 {lon1} != lon1_check {lon1_check}"
    assert lat1 == lat1_check, f"lat1 {lat1} != lat1_check {lat1_check}"

    lon = np.linspace(lon0, lon1, src.width)
    lat = np.linspace(lat0, lat1, src.height)

    dx = lon[1] - lon[0]
    dy = lat[0] - lat[1]
    assert lon[0] - dx/2 == l, f"lon[0] {lon[0]} - dx/2 {dx/2} != l {l}"
    assert lat[0] + dy/2 == t, f"lat[0] {lat[0]} + dy/2 {dy/2} != t {t}"

    return lon, lat

def load_population_count_orig():

    file_path = download_population_count()

    # Open the TIFF file
    with rasterio.open(file_path) as src:
        lon, lat = _get_coords(src)
        data = src.read(1)
        data[data == src.nodata] = np.nan

    pop = xa.DataArray(
        data=data,
        dims=["lat", "lon"],
        coords={
            "lat": lat,
            "lon": lon
        },
        attrs={
            "units": "people",
            "description": "World population count",
            "source": "https://hub.worldpop.org/geodata/summary?id=24777"
        }
    )
    pop.name = "population_count"
    pop.lon.attrs["long_name"] = "Longitude"
    pop.lon.attrs["units"] = "degrees_east"
    pop.lat.attrs["long_name"] = "Latitude"
    pop.lat.attrs["units"] = "degrees_north"

    return pop


def aggregate_population_count(pop, res=0.5, tolerance=0.005, fill_value=np.nan):
    dx = pop.lon.values[1] - pop.lon.values[0]
    dy = pop.lat.values[0] - pop.lat.values[1]
    assert dx.round(6) == dy.round(6), f"dx {dx} != dy {dy}"
    n = int(res / dx)
    if res == 0.5:
        assert n == 60, f"n {n} != 60"
    pop_coarse = pop.coarsen(lon=n, lat=n).sum()

    # also aggregate the mask, unless fill_value is 0
    if fill_value != 0:
        mask = (np.isfinite(pop)).coarsen(lon=n, lat=n).sum().values == 0
        pop_coarse.values[mask] = np.nan

    # True tolerance is 0.01249... -> well below 0.5 degrees
    # print(pop_coarse.lon[:5]) # array([-179.751249, -179.251249, -178.751249, -178.251249, -177.751249])
    # print(pop_coarse.lat[:5]) # array([83.749583, 83.249583, 82.749583, 82.249583, 81.749583])
    lon_coarse = np.arange(-180+res/2, 180, res)
    lat_coarse = np.arange(90-res/2, -90, -res)
    return pop_coarse.reindex(dict(lat=lat_coarse, lon=lon_coarse), tolerance=tolerance, method="nearest", fill_value=fill_value)