"""Regression test for the heating-degree-days RIME-X pipeline.

Compares freshly generated outputs (regional averages + quantile maps)
against checked-in reference outputs.
"""
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from rimeX.datasets.download_isimip import CONFIG, Indicator
from rimeX.preproc.regional_average import run_regional_averages
from rimeX.preproc.quantilemaps import make_quantilemaps

# --------------------------------------------------------------------------
# Paths
#
# Anchored to this file's location (not the current working directory), so
# the test behaves the same whether you run `pytest` from the repo root,
# from tests/regression, or from anywhere else.
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
TUTORIAL_DATA = HERE.parent.parent / "tutorial" / "data"
CREATED_OUTPUT = HERE.parent / "created_output_data"
EXPECTED_OUTPUT = HERE.parent / "expected_output_data"

RAW_DATA_DIR = TUTORIAL_DATA / "raw_data"
MASKS_FOLDER = TUTORIAL_DATA / "continental_masks"
WARMING_LEVEL_FILE = CREATED_OUTPUT / "warming_levels_ISIMIP.csv"

# Filenames encode model, scenario, and the covered time range, e.g.:
# ipsl-cm6a-lr_r1i1p1f1_w5e5_ssp370_hdd-18_global_annual_2015_2100.nc
FILENAME_PATTERN = re.compile(
    r"(?P<model>[a-z0-9\-]+)_r1i1p1f(?P<fnum>\d+)_w5e5_(?P<scenario>[a-z0-9]+)"
    r"_hdd-18_global_annual_(?P<start>\d{4})_(?P<end>\d{4})\.nc$"
)


def _require(path: Path, kind: str = "path") -> Path:
    """Fail early with a clear message instead of a deep, confusing
    FileNotFoundError from inside the pipeline or pandas/xarray."""
    if not path.exists():
        pytest.fail(f"Required {kind} does not exist: {path}", pytrace=False)
    return path


def _build_isimip_db(raw_data_dir: Path, db_file: Path) -> None:
    """Build the ISIMIP metadata db from the raw tutorial .nc files.

    NB: paths inside the db must be absolute -- RIME-X joins them with
    `isimip.download_folder` to check whether the file already exists
    locally; a relative path here would fail that check and trigger an
    (unnecessary, and likely failing) download attempt from the real
    ISIMIP server.
    """
    files = sorted(glob.glob(str(raw_data_dir / "*.nc")))
    if not files:
        pytest.fail(f"No .nc files found in {raw_data_dir}; check the path.", pytrace=False)

    db = []
    unmatched = []
    for f in files:
        m = FILENAME_PATTERN.search(os.path.basename(f))
        if m is None:
            unmatched.append(f)
            continue
        db.append({
            "files": [{"time_slice": [int(m["start"]), int(m["end"])], "path": os.path.abspath(f)}],
            "specifiers": {
                "climate_variable": "heating-degree-days",
                "climate_forcing": m["model"],
                "climate_scenario": m["scenario"],
                "simulation_round": "isimip3b",
                "time_step": "annual",
            },
        })

    # Silently skipping unmatched files hides naming/regex mismatches --
    # fail loudly instead so a typo in the pattern doesn't quietly shrink
    # the test's coverage.
    if unmatched:
        pytest.fail(
            "The following raw data files did not match the expected "
            f"filename pattern and were skipped:\n  " + "\n  ".join(unmatched),
            pytrace=False,
        )
    if not db:
        pytest.fail(f"No files in {raw_data_dir} matched the expected naming pattern.", pytrace=False)

    db_file.parent.mkdir(parents=True, exist_ok=True)
    with open(db_file, "w") as fh:
        json.dump(db, fh, indent=2)


def _assert_csv_matches(output_path: Path, expected_path: Path) -> None:
    _require(expected_path, "expected CSV")
    _require(output_path, "output CSV")
    output = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    pd.testing.assert_frame_equal(output, expected)


def _assert_nc_matches(output_path: Path, expected_path: Path) -> None:
    _require(expected_path, "expected NetCDF")
    _require(output_path, "output NetCDF")
    # Context managers close the files afterwards instead of leaking open
    # file handles for the rest of the test session (xr.load_dataset avoids
    # this too, but open_dataset + `with` is the more explicit/safe form).
    with xr.open_dataset(expected_path) as expected, xr.open_dataset(output_path) as output:
        xr.testing.assert_identical(expected, output)


@pytest.fixture
def hdd_pipeline_config():
    """Point RIME-X's CONFIG at the tutorial test data, build the ISIMIP db,
    and restore the previous CONFIG afterwards so this test doesn't leak
    state into other tests that might run in the same session."""
    _require(RAW_DATA_DIR, "raw data folder")
    _require(MASKS_FOLDER, "continental masks folder")
    _require(WARMING_LEVEL_FILE, "warming level file")

    CREATED_OUTPUT.mkdir(parents=True, exist_ok=True)

    # NOTE: this db is *generated test input*, not ground truth -- it now
    # lives under created_output_data instead of expected_output_data, so
    # nothing here overwrites files in the "expected" reference folder.
    db_file = CREATED_OUTPUT / "heating-degree-days_db.json"
    _build_isimip_db(RAW_DATA_DIR, db_file)

    new_config = {
        "isimip.download_folder": str(CREATED_OUTPUT / "downloads"),
        "indicators.folder": str(CREATED_OUTPUT / "indicators"),
        "isimip.climate_impact_explorer": str(CREATED_OUTPUT),
        "preprocessing.regional.weights": ["latWeight"],
        "preprocessing.regional.masks_folder": str(MASKS_FOLDER),
        "indicator.heating-degree-days": {
            "frequency": "annual",
            "units": "days*°C",
            "isimip_meta": {"db_file": str(db_file)},
        },
    }
    previous = {k: CONFIG.get(k) for k in new_config}
    CONFIG.update(new_config)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                CONFIG.pop(k, None)
            else:
                CONFIG[k] = v


def test_whole_pipeline(hdd_pipeline_config):
    heating_degree_days = Indicator.from_config("heating-degree-days")
    heating_degree_days.run_download(dry_run=False)

    # --- regional averages -------------------------------------------------
    run_regional_averages(["heating-degree-days"], overwrite=True)
    _assert_csv_matches(
        CREATED_OUTPUT / "indicators/heating-degree-days/historical/mpi-esm1-2-hr"
        "/mpi-esm1-2-hr_historical_heating-degree-days_regional_latweight_annual_1850_2014.csv",
        EXPECTED_OUTPUT / "indicators/heating-degree-days/historical/mpi-esm1-2-hr"
        "/mpi-esm1-2-hr_historical_heating-degree-days_regional_latweight_annual_1850_2014.csv",
    )

    # --- regional quantile maps ---------------------------------------------
    make_quantilemaps(
        indicator=["heating-degree-days"],
        quantile_bins=10,
        overwrite=True,
        regional=True,
        warming_level_file=str(WARMING_LEVEL_FILE),
        weight="latWeight",
    )
    _assert_nc_matches(
        CREATED_OUTPUT / "quantilemaps_regional_admin/heating-degree-days/AFR"
        "/heating-degree-days_annual_afr_latweight_qb10-eq.nc",
        EXPECTED_OUTPUT / "quantilemaps_regional_admin/heating-degree-days/AFR"
        "/heating-degree-days_annual_afr_latweight_qb10-eq.nc",
    )

    # --- gridded quantile maps -----------------------------------------------
    make_quantilemaps(
        indicator=["heating-degree-days"],
        map=True,
        quantile_bins=10,
        overwrite=True,
        warming_level_file=str(WARMING_LEVEL_FILE),
    )
    _assert_nc_matches(
        CREATED_OUTPUT / "quantilemaps/heating-degree-days"
        "/heating-degree-days_annual_quantilemaps_qb10-eq.nc",
        EXPECTED_OUTPUT / "quantilemaps/heating-degree-days"
        "/heating-degree-days_annual_quantilemaps_qb10-eq.nc",
    )