# rimeX

## Description

This repository contains code originally written for the [Climate Impact Explorer](https://climate-impact-explorer.climateanalytics.org).
It started as a rewrite of the emulator intended for provide better statistical calculations with exact frequentist estimates.
It was moved to a standalone repository for re-use in various projects, and is intended to supercede the code for the [Rapid Impact Model Emulator](https://github.com/iiasa/rime) (hence its name).


## Back-compatibility and transition period

For users who want to use the original module by Edward Byers instead, the `rimeX.legacy` subpackage is made available.
All `rime` imports were updated with `rimeX.legacy`, but it is otherwise left unedited since import on March 22nd, 2024: `sed -i "s/rime\./rimeX.legacy./g" *.py wip_scraps/*.py`.

It is possible to import via `import rimeX.legacy as rime` to use existing code. Eventually this subpackage will be deprecated.


## Install

A development install can be done after cloning the repo, in pip-editable `-e` mode (that way code edits will propagate without the need for re-installing):

	git clone --single-branch --branch rimeX https://github.com/iiasa/rime.git
	cd rime
	pip install -e .

For the end-user (we're not at this stage yet) or one-off testing,
it's also possible to do it in one go with pip, but the whole repo is cloned in the background so it's slower.
The command is shown below for completeness, but it is not recommended (slower and no edits possible):

 	pip install git+https://github.com/iiasa/rime.git@rimeX


To install all optional dependencies, append `[all]`, e.g. from the local clone:

	pip install -e .[all]


### Quantile maps

![Quantile map](/docs/quantilemap.png)

That's the latest development in rime.

The code is currently contained in [rimeX/preproc/quantilemaps.py](/rimeX/preproc/quantilemaps.py)

- [rimeX.preproc.quantilemaps.make_quantile_map_array](https://github.com/iiasa/rime/blob/rimeX/rimeX/preproc/quantilemaps.py#L30-L265) : create the quantile map as indicated in the picture.
- [rimeX.preproc.quantilemaps.make_quantilemap_prediction](https://github.com/iiasa/rime/blob/rimeX/rimeX/preproc/quantilemaps.py#305-L381) : use the output of `make_quantile_map_array` to make predictions.

(there is also a corresponding command line `rime-pre-quantilemap` that handles the creation of the quantile maps via `make_quantile_map_array`).

See the inline doc `help()` for documentation.


### Command Line Interface

See the [CLI documentation](/docs/cli.md).

Some of these commands might be deprecated in the future, to focus on [quantile maps](#quantile-maps).

## Config files and default parameters

Note the scripts sets default parameters from a [configuration file](rimeX/config.toml).
You can specify your own defaults by having a `rimeX.toml` or `rime.toml` file in the working directory (from which any of the above scripts are called), or by specifying any file via the command-line argument `--config <FILE.toml>`. The `rime-config` script is provided to output the default config to standard output, to save it to file e.g. `rime-config > rime.toml` and later edit `rime.toml` for custom use. Note it is OK to only define a few fields in the config file -- all others will be take from the default config.

If used interactively or imported from a custom script, the config can be changed on-the-fly by accessing the `rimeX.config.CONFIG` flat dictionary.

By default, ISIMIP3b data are used, but that can be changed to ISIMIP2b via the `--simulation-round` flag (available models and experiments and defaults are adjusted automatically).

## Download and pre-processing of new indicators

In addition to ISIMIP variables, a number of indicators are now available which are defined in [config.toml](rime/config.toml).

See [dedicated documentation](docs/preprocessing.md)

The new indicators have been downloaded from the python interactive prompt, such as

```python
from rime.download_isimip import Indicator
indicator = Indicator.from_config("wet_bulb_temperature")
list(indicator.download_all())
```

The command `rime-download-isimip --indicator wet_bulb_temperarure` should also work.