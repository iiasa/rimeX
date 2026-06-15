import glob, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from rimeX.preproc.quantilemaps import make_quantilemap_prediction, make_timesensitive_quantilemap_prediction
from rimeX.emulator import recombine_gmt_ensemble, load_magicc_ensemble
from rimeX.preproc.regional_average import get_all_regions
from rimeX.config import CONFIG
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import copy
from typing import Optional


def _infer_time_aggregation(folder: str) -> str:
    """
    Infer time_aggregation_in from filenames in a given folder.
    Expects filenames matching the pattern *_{aggregation}_{time_agg}_{start}_{end}.csv.
    Returns 'monthly' or 'annual', raises if ambiguous or not found.
    """
    pattern = re.compile(r"_(?P<time_agg>monthly|annual)_\d{4}_\d{4}\.csv$")
    found = set()
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            m = pattern.search(fname)
            if m:
                found.add(m.group("time_agg"))
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        raise ValueError(f"Ambiguous time aggregation in {folder}: {found}")
    raise ValueError(f"Could not infer time_aggregation_in from files in {folder}")


def find_covering_file(
    folder: str,
    model: str,
    scenario: str,
    indicator: str,
    region: str,
    aggregation: str,
    time_aggregation_in: str,
    requested_start: int,
    requested_end: int,
) -> str | None:
    """
    Find a CSV file in *folder* whose year range covers [requested_start, requested_end].

    File names are expected to match:
        {model}_{scenario}_{indicator}_{region}_{aggregation}_{time_aggregation_in}_{start}_{end}.csv

    Returns the full file path of the smallest covering range, or None if no match is found.
    """
    if not os.path.isdir(folder):
        #print(f"[WARN] Folder missing for region '{region}': {folder}")
        return None

    pattern = re.compile(
        rf"{re.escape(model.lower())}_{re.escape(scenario)}_{re.escape(indicator)}"
        rf"_{re.escape(region.lower())}_{re.escape(aggregation)}_{re.escape(time_aggregation_in)}"
        rf"_(\d{{4}})_(\d{{4}})\.csv"
    )

    candidates = [
        (int(m.group(1)), int(m.group(2)), fname)
        for fname in os.listdir(folder)
        if (m := pattern.match(fname))
        and int(m.group(1)) <= requested_start
        and int(m.group(2)) >= requested_end
    ]

    if not candidates:
        return None

    _, _, best_fname = min(candidates, key=lambda x: x[1] - x[0])
    return os.path.join(folder, best_fname)


def load_timeinterval(
    indicator: str,
    model: str,
    scenario: str,
    aggregation: str,
    regions: list[str],
    start_year: int,
    end_year: int,
    province_level: bool = False,
    baseline: tuple | None = None,
    base_path: str = f"{CONFIG['isimip.climate_impact_explorer']}/indicators",
) -> pd.DataFrame:
    """
    Load regional indicator data from pre-computed CSVs, optionally subtract a
    climatological baseline, and return a tidy DataFrame.

    The function automatically infers the temporal resolution of the input data
    (monthly or annual) from the filenames found on disk, and preserves that
    resolution in the output (no additional temporal aggregation is performed).

    Parameters
    ----------
    indicator : str
        Indicator name (e.g. ``"tas"``, ``"pr"``), used to locate files under
        ``base_path/{indicator}/``.
    model : str
        Climate model identifier (case-insensitive folder lookup).
    scenario : str
        SSP/RCP scenario string (e.g. ``"ssp245"``).  If *start_year* < 2015,
        the historical segment (up to 2014) is spliced in automatically.
    aggregation : str
        Spatial aggregation string as encoded in the filenames (e.g.
        ``"national"``, ``"gadm1"``).
    regions : list[str]
        Region identifiers (typically ISO-3 country codes) to load.
    start_year : int
        First year of the requested output period (inclusive).
    end_year : int
        Last year of the requested output period (inclusive).
    province_level : bool, optional
        If ``True``, all sub-national columns in each CSV are retained;
        otherwise only the national-level column (matching *region*) is kept.
        Defaults to ``False``.
    baseline : tuple[int, int] | None, optional
        ``(baseline_start, baseline_end)`` year range for computing a
        climatological mean that is subtracted from the output.  Uses the
        historical scenario files.  Defaults to ``None`` (no anomaly).
    base_path : str, optional
        Root directory that contains per-indicator sub-directories.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by time (format depends on the native
        temporal resolution of the data) with one column per region/province.
        Rows are filtered to [start_year, end_year].

    Raises
    ------
    RuntimeError
        If no data could be loaded for any of the requested regions.
    """

    # ------------------------------------------------------------------
    # Infer time_aggregation_in from disk
    # ------------------------------------------------------------------
    probe_region = regions[0]
    probe_folder = os.path.join(base_path, indicator, scenario, model.lower(), aggregation, probe_region)
    time_aggregation_in = _infer_time_aggregation(probe_folder)

    # ------------------------------------------------------------------
    # Baseline climatology helper
    # ------------------------------------------------------------------
    def _compute_baseline(region: str, col_name: str):
        """
        Return the climatological mean for *col_name* over the baseline period.

        For monthly input the result is a Series indexed by month (1–12).
        For annual input it is a scalar float.
        """
        hist_folder = os.path.join(
            base_path, indicator, "historical", model.lower(), aggregation, region
        )
        hist_path = find_covering_file(
            folder=hist_folder,
            model=model.lower(),
            scenario="historical",
            indicator=indicator,
            region=region,
            aggregation=aggregation,
            time_aggregation_in=time_aggregation_in,
            requested_start=baseline[0],
            requested_end=baseline[1],
        )

        if hist_path is None or not os.path.exists(hist_path):
            print(f"[WARN] Baseline file missing for region '{region}'")
            return None

        df_hist = pd.read_csv(hist_path, parse_dates=["time"], index_col="time")
        if col_name not in df_hist.columns:
            return None

        s = df_hist[col_name]
        s = s[(s.index.year >= baseline[0]) & (s.index.year <= baseline[1])]

        if time_aggregation_in == "monthly":
            # Monthly climatology: mean per calendar month
            return s.groupby(s.index.month).mean().astype(float)
        else:
            # Annual climatology: grand mean scalar
            return float(s.mean())

    # ------------------------------------------------------------------
    # Main loop over regions
    # ------------------------------------------------------------------
    all_series: dict[str, pd.Series] = {}

    for region in regions:
        scenario_folder = os.path.join(base_path, indicator, scenario, model.lower(), aggregation, region)

        if start_year >= 2015:
            file_path = find_covering_file(
                folder=scenario_folder,
                model=model.lower(),
                scenario=scenario.lower(),
                indicator=indicator,
                region=region,
                aggregation=aggregation,
                time_aggregation_in=time_aggregation_in,
                requested_start=start_year,
                requested_end=end_year,
            )
            file_path_hist = None
        else:
            file_path = find_covering_file(
                folder=scenario_folder,
                model=model.lower(),
                scenario=scenario.lower(),
                indicator=indicator,
                region=region,
                aggregation=aggregation,
                time_aggregation_in=time_aggregation_in,
                requested_start=2015,
                requested_end=end_year,
            )
            file_path_hist = find_covering_file(
                folder=os.path.join(base_path, indicator, "historical", model.lower(), aggregation, region),
                model=model.lower(),
                scenario="historical",
                indicator=indicator,
                region=region,
                aggregation=aggregation,
                time_aggregation_in=time_aggregation_in,
                requested_start=start_year,
                requested_end=2014,
            )

        if file_path is None:
            #print(
                #f"[WARN] No covering file found for region '{region}' "
                #f"({indicator}, {scenario}, {model}, {aggregation}, {start_year}–{end_year}) — skipping"
            #)
            continue

        # Load and splice historical + scenario segments when needed
        df = pd.read_csv(file_path, parse_dates=["time"], index_col="time")
        if file_path_hist is not None:
            df_hist_seg = pd.read_csv(file_path_hist, parse_dates=["time"], index_col="time")
            df = pd.concat([df_hist_seg, df]).sort_index()

        # Determine which columns to retain
        if province_level:
            value_cols = list(df.columns)
        else:
            if region not in df.columns:
                #print(f"[WARN] National column '{region}' missing in data — skipping")
                continue
            value_cols = [region]

        for col in value_cols:
            s = df[col].copy()

            # Apply baseline anomaly if requested
            if baseline is not None:
                bval = _compute_baseline(region, col)
                if bval is not None:
                    if time_aggregation_in == "monthly":
                        # Subtract month-specific climatology
                        s = s - s.index.month.map(bval)
                    else:
                        s = s - bval

            # Format the index to a readable string
            if time_aggregation_in == "monthly":
                s.index = s.index.strftime("%Y-%m")
            else:
                s.index = s.index.year

            col_name = col if province_level else region
            all_series[col_name] = s

    if not all_series:
        raise RuntimeError("No data could be processed for any of the requested regions.")

    df_out = pd.DataFrame(all_series).sort_index()

    # Filter to the requested year range
    if time_aggregation_in == "monthly":
        years = pd.to_datetime(df_out.index.astype(str)).year
    else:
        years = df_out.index.astype(int)

    return df_out[(years >= start_year) & (years <= end_year)]

def deduplicate_by_proximity(df, ascending=True, year_diff=4):
    group_cols = ['model', 'impact_model', 'region', 'subregion', 'weights', 'season', 'variable']
    
    df_sorted = df.sort_values('value', ascending=ascending).reset_index(drop=True)
    
    def same_group(r1, r2):
        for c in group_cols:
            v1, v2 = r1[c], r2[c]
            if pd.isna(v1) and pd.isna(v2):
                continue
            if v1 != v2:
                return False
        return True
    
    keep = []
    for idx, row in df_sorted.iterrows():
        dominated = False
        for kept_idx in keep:
            kept_row = df_sorted.loc[kept_idx]
            if same_group(row, kept_row):
                if abs(row['midyear'] - kept_row['midyear']) <= year_diff:
                    dominated = True
                    break
        if not dominated:
            keep.append(idx)
    
    return df_sorted.loc[keep].reset_index(drop=True)

def match_quantiles_to_lookup_sampling(
    scenarios,
    samples,
    lookup_table,
    num_samples=10,
    warming_col="warming_level",
    value_col="value",
    remove_historical = True,
    remove_duplicates = False,
):
    """
    Returns the source simulations of the nearest actually simulated values 
    for a timeseries of samples of impact values for a scenario sampled from 
    a RIME-X distribution
    
    Parameters
    ----------
    scenarios : pd.DataFrame
        Warming levels per year (rows) and model/scenario (columns)
    samples: xr.DataArray
        Samples indexed by year
    lookup_table : pd.DataFrame
        Lookup table containing warming levels and values
    num_samples : int
        Number of rows to return for upper and lower matches
    warming_col : str
        Column name for warming level in lookup_table
    value_col : str
        Column name for values to compare against quantiles
    Returns
    -------
    dict
        {
          year: {
            "lower": [list of row dicts, closest first],
            "upper": [list of row dicts, closest first]
          }
        }
    """
    
    results = {}
    
    for year in samples.year.values:
        
        year = int(year)
        # --- quantile value for this year ---
        q_val = float(samples.sel(year=year).values)
        # --- min / max warming level for this year ---
        year_scenarios = scenarios.loc[year]
        min_wl = round(year_scenarios.min(), 1)
        max_wl = round(year_scenarios.max(), 1)
        # --- filter lookup table to warming level range ---
        lt_filtered = lookup_table[
            (lookup_table[warming_col] >= min_wl) &
            (lookup_table[warming_col] <= max_wl)
        ].copy()

        # Remove historical so that the sample is not used several times for several scenarios
        if remove_historical:
            lt_filtered = lt_filtered[lt_filtered['midyear'].astype(int) >= 2015]
        
        if lt_filtered.empty:
            results[year] = {"lower": [], "upper": []}
            continue
        # --- split into lower / upper relative to quantile ---
        lower_part = lt_filtered[lt_filtered[value_col] <= q_val]
        upper_part = lt_filtered[lt_filtered[value_col] >= q_val]
        # Sort: lower descending (closest to q_val first), upper ascending
        

        if remove_duplicates:
            lower_part = deduplicate_by_proximity(lower_part, ascending=False, year_diff = 4)
            upper_part = deduplicate_by_proximity(upper_part, ascending=True, year_diff = 4)

        lower_count = len(lower_part)
        upper_count = len(upper_part)        
        
        lower_rows = (
            lower_part.nlargest(num_samples, value_col).to_dict(orient="records")
            if not lower_part.empty else []
        )
        upper_rows = (
            upper_part.nsmallest(num_samples, value_col).to_dict(orient="records")
            if not upper_part.empty else []
        )
        # Middle: num_samples nearest neighbours (by absolute distance)
        middle_rows = (
            lt_filtered
            .assign(_dist=lambda d: (d[value_col] - q_val).abs())
            .nsmallest(num_samples, "_dist")
            .drop(columns="_dist")
            .to_dict(orient="records")
        )

        results[year] = {
            "lower":       lower_rows,
            "upper":       upper_rows,
            "middle":      middle_rows,
            "middle_count": num_samples,
            "lower_count": lower_count,
            "upper_count": upper_count,
        }
        
    return results



def reshape_rolling_window(df, N=5):
    """
    Reshapes a time-indexed DataFrame by appending every N consecutive years
    into a single row.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame with years as index
    N : int
        Window size (number of years to append per row)

    Returns
    -------
    pd.DataFrame
        New DataFrame with N * num_columns columns (named 0 to N*num_cols-1),
        indexed by the first year of each window (e.g. 2015, 2020, 2025, ...)
    """
    years = df.index.tolist()
    
    rows = {}
    for i, start_year in enumerate(years[::N]):
        window_years = years[i * N : i * N + N]
        if len(window_years) < N:
            break
        rows[start_year] = df.loc[window_years].values.flatten()
    
    result = pd.DataFrame(rows).T
    result.index.name = "time"
    return result


def sample_timeseries(
    indicator: str,
    region_sampling: str,
    scenario,
    quantile: float,
    regions_emulation: Optional[list] = None,
    subregion_sampling: Optional[str] = None,
    province_level: bool = False,
    weighting_spatial_aggregation: str = "latWeight",
    temporal_aggregation_distribution: Optional[int] = None,
    year_position: str = "mid_year",
    years_per_simulation_piece: Optional[int] = None,
    path_to_quantilemaps: Optional[str] = None,
    number_timeseries: int = 1,
    season: str = "annual",
    equal_model_weighting: bool = True,
    samples: int = 5000,
    skipna: bool = True,
    remove_historical: bool = True,
    remove_duplicates: bool = False,
) -> xr.DataArray:
    """
    Build a spatially and temporally consistent climate impact emulation.

    The function proceeds in four steps:

    1. **Distribution sampling** – a quantile timeseries is drawn from a
       warming-level-conditioned regional indicator distribution stored as a
       quantile-map NetCDF.
    2. **Lookup matching** – for every scenario year the simulation piece
       (model × scenario × midyear triplet) whose indicator value is nearest
       to the sampled quantile is identified from a pre-binned lookup table.
    3. **Data loading** – for each matched piece the full multi-region
       indicator timeseries is loaded with :func:`load_indicator_regions`.
    4. **Assembly** – pieces are concatenated along the time axis and stacked
       into a ``(time, region, member)`` DataArray.

    Parameters
    ----------
    indicator : str
        Indicator name, used to locate quantile-map and lookup-table files and
        to query CONFIG defaults.
    region_sampling : str
        Region from which the quantile timeseries is drawn (usually an
        ISO-3 code or GADM region identifier).
    scenario : pd.DataFrame
        Warming levels per year (rows) and model/scenario (columns).
        Used both to derive warming level ranges for lookup filtering and to
        determine the output time axis.
    quantile : float
        Quantile to track through the indicator distribution, in [0, 1].
    regions_emulation : list[str], optional
        Output regions.  Defaults to all available regions via
        ``get_all_regions()`` if not provided.
    subregion_sampling : str, optional
        Sub-region within *region_sampling* to use for distribution sampling.
        If provided, ``province_level`` is forced to ``True``.
    province_level : bool, optional
        Whether to load province-level data for the emulation output.
    weighting_spatial_aggregation : str, optional
        Spatial aggregation weighting scheme (e.g. ``"latWeight"``).
    temporal_aggregation_distribution : int, optional
        Running-mean window (in years) for the distribution.  Defaults to
        ``CONFIG["preprocessing.running_mean_window"]``.
    year_position : {'mid_year', 'first_year'}
        Whether the matched midyear is the *centre* or the *start* of each
        simulation piece.
    years_per_simulation_piece : int, optional
        Length of each simulation piece.  Inferred from the first two scenario
        years if not provided.
    path_to_quantilemaps : str, optional
        Root path to ISIMIP quantile-map data.  Defaults to
        ``CONFIG["isimip.climate_impact_explorer"]``.
    number_timeseries : int, optional
        Number of parallel emulation members to produce.
    season : str, optional
        Seasonal averaging of the sampled distribution (e.g. ``"annual"``).
    equal_model_weighting : bool, optional
        If ``True`` the quantile map uses equal model weighting.
    samples : int, optional
        Number of Monte Carlo samples used to build the indicator distribution.
    skipna : bool, optional
        Passed through to :func:`make_quantilemap_prediction`.
    remove_historical : bool, optional
        If ``True``, simulation pieces with midyear < 2015 are excluded.
    remove_duplicates : bool, optional
        If ``True``, :func:`deduplicate_by_proximity` is applied before
        selecting the nearest neighbours.

    Returns
    -------
    xr.DataArray
        Shape ``(time, region, member)`` containing the stitched emulation.
        The time coordinate is inherited from the loaded simulation pieces
        (monthly or annual depending on the native data resolution).
    """
    # ------------------------------------------------------------------
    # Resolve defaults from CONFIG
    # ------------------------------------------------------------------
    if temporal_aggregation_distribution is None:
        temporal_aggregation_distribution = CONFIG["preprocessing.running_mean_window"]
    if path_to_quantilemaps is None:
        path_to_quantilemaps = CONFIG["isimip.climate_impact_explorer"]
    if regions_emulation is None:
        regions_emulation = get_all_regions()

    subregion = subregion_sampling if subregion_sampling is not None else region_sampling
    if subregion != region_sampling:
        province_level = True

    transform = CONFIG.get(f"indicator.{indicator}.transform", "")
    if "baseline" in transform:
        y1, y2 = CONFIG.get(
            f"indicator.{indicator}.projection_baseline",
            CONFIG["preprocessing.projection_baseline"],
        )
        baseline_string = f"_baseline-{y1}-{y2}"
        baseline = (int(y1), int(y2))
    else:
        baseline_string = ""
        baseline = None

    sorted_years = sorted(int(y) for y in scenario.index)
    if years_per_simulation_piece is None:
        years_per_simulation_piece = sorted_years[1] - sorted_years[0] if len(sorted_years) >= 2 else 1

    mid_year_mode = (year_position == "mid_year")

    # ------------------------------------------------------------------
    # Step 1 – load quantile map and sample quantile timeseries
    # ------------------------------------------------------------------
    eq_suffix = "_eq" if equal_model_weighting else ""
    quantilemap_path = (
        f"{path_to_quantilemaps}/isimip3"
        f"/running-{temporal_aggregation_distribution}-years"
        f"/quantilemaps_regional_admin/{indicator}/{region_sampling}"
        f"/{indicator}_{season}_{region_sampling.lower()}"
        f"_{weighting_spatial_aggregation}{eq_suffix}.nc"
    )

    with xr.open_dataset(quantilemap_path) as ds:
        quantilemap = ds[indicator].sel(region=subregion).load()

    # quantile_timeseries shape: (year, quantile); extract the single quantile
    # as a (year,) DataArray — this is what match_quantiles_to_lookup_sampling
    # expects as its `samples` argument
    quantile_timeseries = make_quantilemap_prediction(
        quantilemap, scenario, samples=samples, quantiles=[quantile], skipna=skipna
    )
    quantile_values = quantile_timeseries.sel(quantile=quantile)  # shape: (year,)

    # ------------------------------------------------------------------
    # Step 2 – load lookup table and match quantiles to simulation pieces
    # ------------------------------------------------------------------
    lookup_table_path = (
        f"{path_to_quantilemaps}/isimip3"
        f"/running-{temporal_aggregation_distribution}-years"
        f"/isimip_binned_data/{indicator}/{region_sampling}/{subregion}"
        f"/{weighting_spatial_aggregation}"
        f"/{indicator}_{region_sampling.lower()}_{subregion.lower()}"
        f"_{season}_{weighting_spatial_aggregation.lower()}"
        f"_{temporal_aggregation_distribution}-yrs{baseline_string}.csv"
    )
    lookup_table = pd.read_csv(lookup_table_path)

    out_dict = match_quantiles_to_lookup_sampling(
        scenarios=scenario,            # pd.DataFrame: year × model/scenario
        samples=quantile_values,       # xr.DataArray with year dim
        lookup_table=lookup_table,
        num_samples=number_timeseries, # renamed from number_timeseries
        warming_col="warming_level",
        value_col="value",
        remove_historical=remove_historical,
        remove_duplicates=remove_duplicates,
    )

    # ------------------------------------------------------------------
    # Step 3 – load indicator data for each matched simulation piece
    # ------------------------------------------------------------------
    simulation_parts: list[list[pd.DataFrame]] = []

    #for year in sorted_years:
    #    pieces = out_dict.get(year, {})
    #    year_dfs: list[pd.DataFrame] = []

     #   for sample_idx in range(pieces.get("middle_count", 0)):
     #       match = pieces["middle"][sample_idx]
     #       mid_year = match["midyear"]

      #      if mid_year_mode:
      #          half = (years_per_simulation_piece - 1) // 2
     #           piece_start = mid_year - half
      #          piece_end   = mid_year + (years_per_simulation_piece // 2)
      #      else:
       #         piece_start = mid_year
      #          piece_end   = mid_year + years_per_simulation_piece - 1

       #     df_piece = load_timeinterval(
       #         indicator=indicator,
       #         model=match["model"],
       #         scenario=match["scenario"],
       #         aggregation=weighting_spatial_aggregation,
       #         regions=regions_emulation,
       #         start_year=piece_start,
       #         end_year=piece_end,
       #         province_level=province_level,
       #         baseline=baseline,
       #         base_path=os.path.join(path_to_quantilemaps, "indicators"),
       #     )
        #    year_dfs.append(df_piece)

        #simulation_parts.append(year_dfs)
    for year in sorted_years:
        pieces = out_dict.get(year, {})
        year_dfs: list[pd.DataFrame] = []
    
        for sample_idx in range(pieces.get("middle_count", 0)):
            match = pieces["middle"][sample_idx]
            mid_year = match["midyear"]
    
            if mid_year_mode:
                half = (years_per_simulation_piece - 1) // 2
                piece_start = mid_year - half
                piece_end   = mid_year + (years_per_simulation_piece // 2)
            else:
                piece_start = mid_year
                piece_end   = mid_year + years_per_simulation_piece - 1
    
            df_piece = load_timeinterval(
                indicator=indicator,
                model=match["model"],
                scenario=match["scenario"],
                aggregation=weighting_spatial_aggregation,
                regions=regions_emulation,
                start_year=piece_start,
                end_year=piece_end,
                province_level=province_level,
                baseline=baseline,
                base_path=os.path.join(path_to_quantilemaps, "indicators"),
            )
            # ── remap time axis to be centered on `year` ──────────────────
            if mid_year_mode:
                new_start = year - half
            else:
                new_start = year
            df_piece.index = np.arange(new_start, new_start + len(df_piece))
        # ──────────────────────────────────────────────────────────────
            year_dfs.append(df_piece)

        simulation_parts.append(year_dfs)
    # ------------------------------------------------------------------
    # Step 4 – assemble into (time, region, member) DataArray
    # ------------------------------------------------------------------
    n_members = min(
        (len(year_dfs) for year_dfs in simulation_parts if year_dfs),
        default=0,
    )
    if n_members == 0:
        raise RuntimeError("No simulation pieces could be loaded.")

    member_timeseries: list[pd.DataFrame] = []
    for member_idx in range(n_members):
        pieces_for_member = [
            year_dfs[member_idx]
            for year_dfs in simulation_parts
            if member_idx < len(year_dfs)
        ]
        combined = pd.concat(pieces_for_member, axis=0)
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        member_timeseries.append(combined)

    arrays = []
    for member_idx, df in enumerate(member_timeseries):
        da = xr.DataArray(
            df.values,
            dims=["time", "region"],
            coords={"time": df.index.values, "region": df.columns.values},
        )
        arrays.append(da)

    full_emulation = (
        xr.concat(arrays, dim="member")
        .assign_coords(member=list(range(n_members)))
        .transpose("time", "region", "member")
    )

    return full_emulation

    

    



    