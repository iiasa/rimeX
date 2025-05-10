# Emulator

## Command-line interface

An extensive command-line interface was developped with the objective of back-compatibility with the original `rime` in mind,
and is documented in the [run](/docs/run.md) section. However, this currently does not rely on the quantile map approach.

The recommended usage of `rime` is now via the quantile maps, which is not yet integrated into the `rime-run-...` commands (it will be eventually).

## Quantile maps

![Quantile map](/docs/quantilemap.png)

To use the quantile maps described in [preprocessing](/docs/preprocessing.md), the Climate Impact Explorer uses the following:

```python
import pandas as pd
import xarray as xa
from rimeX.preproc.quantilemaps import make_quantilemap_prediction, get_filepath

# gmt loaded as a dataframe time as index x ensemble as columns
gmt = pd.read_csv(...)

# specify indicator
indicator_name = "rx5day"
region = "ITA"
subregion = "ITA"
season = "annual"
weight = "latWeight"

# the suffix corresponds to equal model weights
fp = get_filepath(indicator_name, season=season, region=region, regional_weights=weight, suffix="_eq")

with xa.open_dataset(fp) as ds:
	impact_data = ds[indicator_name].sel(region=subregion).load()

results = make_quantilemap_prediction(impact_data, gmt, quantiles=[0.5, .05, .95], samples=5000, clip=True, seed=42)

results.T.to_pandas().to_csv("cie_rx5day.csv")
```

See the inline doc `help()` for full documentation.
See also the doc for [preprocessing](/docs/preprocessing.md#quantile-maps)