# Pre-processing

Rime can handle indicator pre-processing, provided the [`config.toml` file](/README.md#config-files-and-default-parameters) is updated.
Each indicator should provide enough metadata to be identified unequivocally on the ISIMIP database or on the local disk.

## ISIMIP variables: basic climate model output

The simple direct outputs from climate models do not need any specific description. It is assumed that the data should be aggregted to monthly outputs, and that besides `variable`, the only specifers are `climate_forcing` (the ESM, e.g. "GFDL-ESM4"), `climate_scenario` (e.g. "SSP5-8.5"). If the `frequency` on the indicator should be different from `monthly`, this should be indicated (`daily` or `annual` are supported). It is possible to restrict the country averaging to only specific spatial weights for a given indicator (by default all three weight types indicated below will be processed). The variables are often expressed relatively to a baseline. This is specified via the `transform` keyword. Already implemented tranformations are `baseline_change_percent` (e.g. precipitation) and `baseline_change` (temperature). By default no transform is applied. Below is the default, with a summary of all available options commented out:

```toml
[indicator.<name>]
frequency = "monthly"
# frequency = "daily"
# frequency = "annual"
spatial_aggregation = ["latWeight", "pop2020", "gdp2020"]
# no transform by default, alternatives are
# transform = "baseline_change"
# transform = "baseline_change_percent"
# projection_baseline = ...  # indicates if differs from global defaults
# units = ...  # write to netCDF, figures, documentation
# comment = ... # documentation
# title = ...  # for figures, documentation
#
# depends_on = [...]  # see below
# depends_on_climatology = true
# climatology_quantile = 0.999
# expr = ...
# shell = ...
# custom = ...
```

## ISIMIP variables: derived outputs

Derived outputs may have several datasets per climate model and scenario, e.g. crop or hydrological models require an additional `model` field. That should be indicated via an additional `isimip_meta` descriptor, which is provided as an additional group. Example of the Maize yield, whose name `maize_yield` does not exist on the ISIMIP database:

```toml
[indicator.maize_yield]
frequency = "annual"
spatial_aggregation = ["latWeight"]
units = "%"

[indicator.maize_yield.isimip_meta]
specifiers = ["variable", "crop"]
variable = "yieldchange"
crop = "mai"
ensemble_specifiers = ["model"] # further specifiers to identify the ensemble member
```

A mapping to the ISIMIP database is provided in `isimip_meta` by using `variable` and `crop` specifiers, with the values `variable="yieldchange"` and `crop="mai"`. Note lists are also accepted to select several values for those specifiers, but it is expected that only one data entry will be found after filtering with all specifiers, and will raise an error otherwise. We also want all crop models to be considered as part of our ensemble (it will be added to the default "ensemble" specifiers `climate_forcing` and `climate_scenario`). See below the precise formatting.

The entries fetched from the ISIMIP database will be saved to the ISIMIP download folder (see `[isimip]` section in the config file) under a `db` subfolder.
Later on, you might explicitly specify to read the metadata from that file instead of fetch from the internet, here `db_file = "db/maize_yield.json"`, to enforce reproductibility as the ISIMIP database is expanded. See below how that the `db_file` field can be used to create new, custom indicators.


## Custom variables derived from ISIMIP variables

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

This makes use of a custom python function, here defined in a rime submodule [rimeX.indicators](/rimeX/indicators.py).
The function is called on each time-slice file and passed the unnamed parameters (*args):

- input_daily_files : list of files, must be the same length of `depends_on`
- [clim_files] : [only if depends_on_climatology is True] netcdf files containing the climatological mean (or quantile if `climatology_quantile` is passed) calculated on-the-fly by rime, for each of the input variables, same length as `depends_on` (so far it was only used for monovariate indicators)
- time_slice_file: the output netcdf file for this time slice

and named parameters (**kwargs):

- previous_input_files: input files from previous time slices (useful to **accumulate** quantities, e.g. precipitation in `rx5day`)
- previous_output_file: output files from previous time slices
- dry_run: don't actually do anything (if implemented in the custom function), just for developping

or in a nutshell:
```python
    if depends_on_climatology:
        func(input_daily_files, clim_files, time_slice_file, **kwargs)
    else:
        func(input_daily_files, time_slice_file, **kwargs)
```

Besides `expr` and `custom`, also implemented (but not used / tested so far) is a `shell` keyword, which is a shell command with placeholders to be formatted with `str.format()`
```
{inputs} : a list of input files
{input} : joined inputs with " " separator
{output} : the output file
{previous_inputs} : {inputs} from the previous time slice
{previous_input} : joined {previous_inputs} with " " separator
{previous_output} : the first element of {previous_outputs}
{name}
```


## Custom indicator not downloaded from ISIMIP

See the `wsi` entry in [config.toml](/rimeX/config.toml) and the corresponding [examples/wsi.json](/examples/wsi.json) db file:

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
