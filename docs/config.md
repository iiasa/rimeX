# Config file

Note the scripts sets default parameters from a [configuration file](/rimeX/config.toml).

You can specify your own defaults by having a `rimeX.toml` or `rime.toml` file in the working directory (from which any of the above scripts are called), or by specifying any file via the command-line argument `--config <FILE.toml>`. The `rime-config` script is provided to output the default config to standard output, to save it to file e.g. `rime-config > rime.toml` and later edit `rime.toml` for custom use. Note it is OK to only define a few fields in the config file -- all others will be take from the default config.

If used interactively or imported from a custom script, the config can be changed on-the-fly by accessing the `rimeX.config.CONFIG` flat dictionary.

By default, ISIMIP3b data are used, but that can be changed to ISIMIP2b via the `--simulation-round` flag (available models and experiments and defaults are adjusted automatically).

Since the config file contains settings across all rime functionalities, its content is covered throughout the docs:

- [indicator specification](/docs/indicators.md)

