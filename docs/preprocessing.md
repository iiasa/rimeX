# Pre-processing

Rime can handle indicator pre-processing, provided the [`config.toml` file](../README.md#config-files-and-default-parameters) is updated.
Each indicator should provide enough metadata to be identified unequivocally on the ISIMIP database or on the local disk.

## ISIMIP variables

The simple direct outputs from climate models do not need any specific description. It is assumed that the data should be aggregted to monthly outputs, and that besides `variable`, the only specifers are `climate_forcing` (the ESM, e.g. "GFDL-ESM4"), `climate_scenario` (e.g. "SSP5-8.5"). If the frequency on the indicator should be different from monthly, this should be indicated (`daily` or `annual` are supported).

```toml
[indicator.maize_yield]
frequency = "annual"
spatial_aggregation = ["latWeight"]
units = "%"
```
Additionally, for that indicator we only want area-based regional average (not interested in population or GDP weighting), so we additionally specify that. In this case, we additionally document that the units in `%` (and not further transformation will be apply). Other variables may require specific transform via the `transform` keyword next to `frequency`. Already implemented tranformations are `baseline_change_percent` (e.g. precipitation) and `baseline_change` (temperature).

Derived outputs may have several datasets per climate model and scenario, e.g. crop or hydrological models require an additional `model` field. That should be indicated via the `isimip_meta` descriptor. Example of the Maize yield, whose name `maize_yield` does not exist on the ISIMIP database:

```toml
[indicator.maize_yield.isimip_meta]
specifiers = ["variable", "crop"]
variable = "yieldchange"
crop = "mai"
ensemble_specifiers = ["model"] # further specifiers to identify the ensemble member
```

A mapping to the ISIMIP database is provided in `isimip_meta` by using `variable` and `crop` specifiers, with the values `variable="yieldchange"` and `crop="mai"`. Note lists are also accepted to select several values for those specifiers, but it is expected that only one data entry will be found after filtering with all specifiers, and will raise an error otherwise. We also want all crop models to be considered as part of our ensemble (it will be added to the default "ensemble" specifiers `climate_forcing` and `climate_scenario`). See below the precise formatting.

The entries fetched from the ISIMIP database will be saved to the ISIMIP download folder (see `[isimip]` section in the config file) under a `db` subfolder.
Later on, you might explicitly specify to read the metadata from that file instead of fetch from the internet, here `db_file = "db/maize_yield.json"`, to enforce reproductibility as the ISIMIP database is expanded. See below how that the `db_file` field can be used to create new, custom indicators.


## Derived from ISIMIP variables

Some variables may be derived from other variables. There is a machinery to do some basic transformations. Let's go through existing indicators:
```toml
[indicator.wet_bulb_temperature]
depends_on = ["tasmax", "hurs"]
expr = "(tasmax-273.15)*atan(0.151977*(hurs+8.313659)^0.5) + atan((tasmax-273.15)+hurs) - atan(hurs-1.676331) + 0.00391838*hurs^1.5*atan(0.023101*hurs) - 4.686035"
```
The `depends_on` field is essential for all derived indicators. The code will fetch the metadata for each precursor (available models, scenarios, etc) and build a database with the intersection of those. The `expr` field is passed to `cdo expr` command.

A more complex example:
```toml
[indicator.extreme_daily_rainfall]
depends_on = ["pr"]
depends_on_climatology = true
climatology_quantile = 0.999
custom = "rimeX.indicators:extreme_daily_rainfall"
```
This makes use of a custom python function, here defined in a rime submodule [rimeX.indicators](rimeX/indicators.py).
See that file for details.


## Custom indicator not downloaded from ISIMIP

See the `wsi` entry in [config.toml](rimeX/config.toml) and the corresponding [examples/wsi.json](examples/wsi.json) db file:

```toml
[indicator.wsi.isimip_meta]
db_file = "examples/wsi.json"
ensemble_specifiers = ["model"]
```

The example database contaisn a single file. The path is relative to the isimip download folder but can be also provided as absolute path.

```json
[
    {
        "files": [{
            "time_slice": [1980, 2100],
            "path" : "wsi/wsi-gfdl-esm4_impact1_historical.nc"
        }
        ],
        "specifiers": {
            "climate_variable" : "wsi",
            "climate_forcing" : "GFDL-ESM4",
            "climate_scenario" : "ssp585",
            "model" : "impact1",
            "simulation_round" : "isimip3b"
        }
    }
]
```