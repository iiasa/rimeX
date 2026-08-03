# rimeX

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

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
pip install -e '.[all]'
```

### Conda installation (alternative)

If you prefer using conda, you can create an environment and install dependencies as follows:

```bash
conda create -n rimex-env python=3.10
conda activate rimex-env
conda install -c conda-forge cdo
cd rimeX
pip install -e .
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

## License

rimeX is licensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later) — see [LICENSE](LICENSE).

Component provenance and attribution, including code originating from [iiasa/rime](https://github.com/iiasa/rime),
is recorded in [NOTICE](NOTICE).

## Commercial licensing

AGPL-3.0 permits free use, modification and redistribution, provided that
redistributed or network-hosted derivative works are also released under
AGPL-3.0.

Organisations that wish to distribute rimeX, or offer it as a hosted service,
without releasing their own modifications under AGPL-3.0 may obtain a
commercial licence. Contact rimex@iiasa.ac.at for terms.

**Do I need a commercial licence?**

- **No** — if you use rimeX to produce reports, analyses or other deliverables
  for clients, without distributing or hosting rimeX itself.
- **No** — if you run rimeX internally within your organisation, however
  modified, without redistributing it.
- **Yes** — if you distribute rimeX (modified or not) as part of a proprietary
  product, or run a modified version as a network-accessible service, and wish
  to keep your modifications closed.

## Citation

If you use rimeX in any published work, please cite:

> Schwind, N., Perrette, M., Byers, E., Högner, A., Lejeune, Q., Möller, T., Nicholls, Z., Pfleiderer, P., Schöngart, S., Werning, M., and Schleussner, C.-F.: RIME-X v1.0: combining simple climate models, Earth system models, and climate impact models into a unified statistical emulator for regional climate indicators, Geosci. Model Dev., 19, 6797–6828, https://doi.org/10.5194/gmd-19-6797-2026, 2026.

and the archived software version (DOI: [10.5281/zenodo.17491734](https://doi.org/10.5281/zenodo.17491734)). Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

If your use of rimeX leads to a substantial scientific contribution, we would
welcome a conversation about collaboration or co-authorship — please get in touch.
We are keen to keep track of its use.

## Data sources

rimeX downloads and processes third-party climate data, including ISIMIP output
and the datasets of Werning et al. (2024). The rimeX licence covers the software
only. Data obtained through the download scripts remains subject to its own terms
of use and attribution requirements.

## Versions

rimeX follows semantic versioning. v1.0.0 is the version described in
Schwind et al. (2026) and archived at [10.5281/zenodo.17491734](https://doi.org/10.5281/zenodo.17491734).

Licence terms for future releases may differ. v1.0.0 remains available under
AGPL-3.0-or-later permanently.
