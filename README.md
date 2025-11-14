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

```bash
git clone https://github.com/iiasa/rimeX.git
cd rimeX
pip install -e .
```

If you need to run the **optional preprocessing yourself**, you also need to install **CDO**.
This can be done system-wide (e.g. using `apt-get` or `brew`) or via conda:

```bash
# Linux
sudo apt-get install cdo

# macOS
brew install cdo

# Or using conda (works everywhere)
conda install -c conda-forge cdo
```

For the end-user (we're not at this stage yet) or one-off testing, it's also possible to do it in one go with pip, but the whole repo is cloned in the background so it's slower.
The command is shown below for completeness, but it is not recommended (slower and no edits possible):

```bash
pip install git+https://github.com/iiasa/rimeX.git
```

To install all optional dependencies, append `[all]`, e.g. from the local clone:

```bash
pip install -e .[all]
```

### Conda installation (alternative)

If you prefer using conda, you can create an environment and install dependencies as follows:

```bash
conda create -n rimex-env python=3.10
conda activate rimex-env
conda install -c conda-forge cdo
pip install -e 
```

## Usage

The `rimeX` package contains relatively distinct functionality, which can be split between [pre-processing](/docs/preprocessing.md) and the emulator itself.
Much of it has a command-line interface, with the notable exception of the latest quantile maps, which is only implemented as python API (see below).
Here is an index of the documentation:

- [config](/docs/config.md) : how to have your own, discoverable config.toml file
- [preprocessing](/docs/preprocessing.md) : how to define new indicators, calculate regional averages, global mean and prepare emulator data (quantile maps)
	- [indicators](/docs/indicators.md) : update the config.toml file for new indicators
- [emulator](/docs/emulator.md) : use the emulator
	- [run](/docs/run.md) : command-line interface to run the emulator (EXPERIMENTAL) -- does not include quantile maps


![Quantile maps](/docs/quantilemap.png)

### Command Line Interface

The following scripts are made available, for which inline help is available with `-h` or `--help`:

- Data download and pre-processing scripts (presently ISIMIP only, variables tas and pr, written for the CIE dataset and masks)

	- `rime-download-isimip` : download ISIMIP data
	- `rime-download` : download other datasets (Werning et al 2024) etc. (platform-independent)
  	- `rime-pre-gmt` : pre-processing: crunch global-mean-temperature
	- `rime-pre-region` : pre-precessing: crunch regional averages (=> this currently requires Climate Impact Explorer masks)
	- `rime-pre-wl` : crunch the warming levels
	- `rime-pre-digitize` : pre-compute digitized regional average based on warming levels (optional -- DEPRECATED)
	- `rime-pre-quantilemap` : produce quantile maps (after running rime-pre-gmt, rime-pre-region and rime-pre-wl)

- Actually use the emulator (works anywhere as long as the data is available) -- EXPERIMENTAL

	- `rime-run-timeseries` : (OUT OF DATE IN ITS CURRENT FORM -> should be replaced with QUANTILE MAP approach) run the main emulator with proper uncertainty calculations (time-series)
	- `rime-run-table` : vectorized version of `rime-run-timeseries` with on-the-fly interpolation, without uncertainties recombination
	- `rime-run-map` : run the map emulator

- Also useful to specify the data paths:

	- `rime-config` : print the config to screen (toml format)
