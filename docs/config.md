# Config file

Note the scripts sets default parameters from a [config.tom](/rimeX/config.toml) configuration file.

You can specify your own defaults by having a `rimeX.toml` or `rime.toml` file in the working directory (from which any of the above scripts are called), or by specifying any file via the command-line argument `--config <FILE.toml>`. 

To check if `rime` discovers you file as expected, or just checks the defaults, you can use the command:
```bash
rime-config
```

The `rime-config` script is provided to output the default config to standard output. It can easily be saved to file e.g. `rime-config > rime.toml` and later edited `rime.toml` for custom use. Note it is OK to only define a few fields in the config file -- all others will be take from the default config.

If used interactively or imported from a custom script, the config can be changed on-the-fly by accessing the `rimeX.config.CONFIG` flat dictionary:
```python
from rimeX.config import CONFIG
CONFIG["isimip.simulation_round"] = [ "isimip3b" ]
```

Since the config file contains settings across all rime functionalities, its content is covered throughout the docs, mostly in the preprocessing sections:

- [preprocessing](/docs/preprocessing.md)
- [indicator specification](/docs/indicators.md)
