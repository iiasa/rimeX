# Preprocssing

## Define indicators

In addition to ISIMIP variables, additional indicators can be defined in [config.toml](/rimeX/config.toml)
See:
(or an updated version of the file located and named according to ["config"](/docs/config.md)),
following rules explained in ["indicators"](/docs/indicators.md).

Once specified, the corresponding indicator class can be defined, for example:

```python
from rime.download_isimip import Indicator
indicator = Indicator.from_config("wet_bulb_temperature")
```

The new indicators have been downloaded from the python interactive prompt, such as

```python
list(indicator.download_all())
```
(`list` because the function returns an iterator)

or more commonly via the command line:

```bash
rime-download-isimip --indicator wet_bulb_temperarure
```

By default all models and all scenarios will be downloaded for the simulation round(s) selected, and according to the configuration file.
This can be overwritten by command line arguments `--model`, `--impact-model`, `--experiment`, `--simulation-round`, which all accept several values.
Daily files are first downloaded, then monthly averages are computed (if applicable). If disk space is scarse, or if you won't re-use the daily files for other indicators,
you can pass `--remove-daily` for final cleanup. The files will be downloaded in the config variable `isimip.download_folder`, by default
```toml
[isimip]
download_folder = "/mnt/ISIMIP"
```
To see complete usage information:

```bash
rime-download-isimip --help
```

## Calculate regional averages

To calculate regional (in general, country) averages, several regional weights can be used, either geographic (`latWeight`), population `pop2020`, GDP (`gdp2020`).
However the paths to the masks is currently hard-coded and expected to follow a certain structure stemming from the Climate Impact Explorer.
The only option is to specify the root path
```
[preprocessing.regional]
weights = [ "latWeight", "gdp2020", "pop2020"]
masks_folder = "/mnt/PROVIDE/climate_impact_explorer/data/masks"
```
and the masks will be looked in `f"{masks_folder}/{region}/masks/{region}_360x720lat89p75to-89p75lon-179p75to179p75_{weights}.nc4"`
At the time of writing, the masks files are extracts from a 0.5 degree grid as suggested by the file name, but restricted to the bounding
box for the specific region. The indicator data is re-indexed to match that grid (in the pandas / xarray sense).
Note the indicator data is not currently interpolated, so the mask and indicator files should
be defined on that same grid or the re-indexing will yield NaNs. The mask netcdf dataset can have several variables, which correspond to subregions.
E.g for Italy, the region will be `ITA`, and the netCDF will contain
one `ITA` variable for the whole country, and then other variables for the Italian regions.

```bash
rime-pre-region --indicator wet_bulb_temperature --weights latWeight gdp2020 pop2020 --region ITA DEU
```

The regional averages are written according to the config variable `indicators.folder`, by default
```toml
[indicators]
folder="/mnt/ISIMIP/indicators"
```

## Prepare the warming level table

The next step is to compute global-mean `tas` and scan all available simulations to be binned into warming levels.
This is achieved with the commands `rime-pre-gmt` and `rime-pre-wl`, respectively.

The warming level file is saved under the folder specified by `isimip.climate_impact_explorer`, by default:
```toml
[isimip]
climate_impact_explorer = "/mnt/PROVIDE/climate_impact_explorer"
```
then a specific subfolder name is built based on the selected options `preprocessing.running_mean_window` (default=21), `isimip.simulation_round` and `preprocessing.tag`, if any.

Full options available with:

```bash
rime-pre-gmt --help   # calculate global mean temperature
rime-pre-wl --help    # prepare the warming level table
```


## Quantile maps

The recommended approach is to combine regional averages with warming level tables is to use quantile maps.

![Quantile map](/docs/quantilemap.png)

Quantile maps are 2-D (+ any other dimensions) arrays
indexed by warming levels and quantiles, i.e. describing the probability distribution of the indicator for each warming level.
Because of the structure of netCDF datasets and corresponding xarray calculations, the quantile maps can have any trailing dimensions
such as geographical coordinates (maps) or regions or/and their administrative boundaries.

A command `rime-pre-quantilemap` is available to produce the quantile maps. Three kinds of outputs are available:
- `--regional` : create one netcdf file per region, that includes all administrative sub-regions
```bash
rime-pre-quantilemap --region ITA --season annual summer winter spring autumn --regional--quantile-bins 101 --weights latWeight gdp2020 pop2020
```
- `--regional-no-admin` : create one netcdf file that includes all regions, without adminsitritive sub-regions
```bash
rime-pre-quantilemap --season annual summer winter spring autumn --regional-no-admin --quantile-bins 101 --weights latWeight gdp2020 pop2020
```
- `--map` : create one netCDF file with geographical coordianates (lon/lat grid points), covering the whole world.
```bash
rime-pre-quantilemap --season annual summer winter spring autumn --map --map-chunk-size 60 --quantile-bins 11 --warming-levels 1.5 2 2.5 3 3.5 4
```
The latter, map output, is quite large and generally used for illustrative purpose only,
so we generally limit the number of quantiles and warming levels to keep memory usage in check. An odd quantile bin number includes the median.
The main warming level file will be loaded and only the required warming levels are selected.

The defaults are provided as in the config file as:
```toml
[preprocessing]
warming_level_step = 0.1  # used for pre-processing (input files for emulator)
warming_level_min = 1
running_mean_window = 21
isimip_binned_backend = [ "csv", ] # "feather" and "parquet" are also available
projection_baseline = [ 1995, 2014,] # this is used in rime-preproc-digitize
quantilemap_quantile_bins = 101
```

Full documentation available at:

```bash
rime-pre-quantilemap --help
```

## Deprecated

The `rime-pre-digitize` command is an alternative pre-processing step
that counts all data points in the warming level bins and write them to a CSV file.
This is an optional pre-processing step for the `rime-run-timeseries` command.
That command is deprecated and eventually `rime-run-timeseries` will be based on quantile maps.