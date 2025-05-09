## TODO (This list dates back June 2024 and needs update, but is kept here for reference)

### Internal classes (maybe ...)

A prospective API would be a set of classes with methods, some of which would be shared across `rime-run-timeseries` and `rime-run-table`. The proposal below has a focus on the internal data structure.

- `ImpactRecords` -> internal data structure is a list of records: this is the base structure of `rime-run-timeseries`. The methods below return another `ImpactRecords` instance, unless otherwise specified:
	- interpolate_years
	- interpolate_warming_levels
	- mean : (current average_per_group)
	- make_equiprobable_groups
	- resample_from_quantiles
	- resample_dims : resample from an index (e.g. model, scenario)
	- resample_montecarlo : return an `ImpactEnsemble` instance
	- to_frame() (internally `ImpactFrame(pandas.DataFrame(self.records))`) would return an `ImpactFrame`
	- sample_by_gmt_pathway() -> return a DataFrame of results (current `recombine_gmt_ensemble`)

- `ImpactEnsemble` : DataFrame with an index (years as multi-index if other dimensions should be accounted for), and samples as columns, for vectorized sampling.
	- sample_by_gmt_pathway() -> return a DataFrame of results (current `recombine_gmt_vectorized`)

- `ImpactFrame`: semantically equivalent to `ImpactRecords`, but internal data structure is a DataFrame : good for reading, writing and as an intermediate state, but operations of destruction / reconstruction can be costly. Eventually, this could be merged with `ImpactRecords`, with one or ther other taking over depending on performance tests.
For now `ImpactRecords` is the main class for work, and `ImpactFrame` is mostly a data holder.
	- ... : some of the methods above may also be implemented using pandas methods (should be equivalent to ImpactRecords methods: ideal for unit tests)
	- to_cube(dims) -> transform to `ImpactCube`

- `ImpactCube` : current `ImpactDataInterpolator`, whose internal data structure is a DataArray
	- sel : (select e.g. specific SSP family)
	- interpolate_by_warming_levels(warming_levels)
	- interpolate_by_warming_levels_and_year()
	- to_frame() -> back to ImpactFrame, which can be more suitable for certain operations


### Scripts

In general: harmonize `rime-run-timeseries` and `rime-run-table`. The former should be able to do much of what the latter can do (except on-the-fly interp):

- `rime-run-table`: only match SSP and `year` on-demand
- `rime-run-timeseries`: add do not mix everything by default: use groupby (and mix on demand)
- both: pool [+ mean or other stat] scenario / years / models before interp


NOTE about `rime-run-table` ssp-family indexing:

- currently we pool SSP_family in impact data (mapped from scenarios), but for the above example there is not reason to do that

Currently we assume a set of "coordinates" dimensions to keep track of (`model, scenario, quantile, variable, year, warming_level, ...`) and to be passed to groupby. It's probably best to pass an `--index` parameter to specify what dimensions should be considered for indexing, groupby etc. : WE NOW DO THIS FOR `rime-run-timeseries`.

We currently "guess" some fields (lower case, remove space and hyphen, and even rename a few). Possibly use an explicit mapping as user input for renaming to standard fields without the current guessing.


### ADD UNIT TESTS
### Rename rimeX to rime
