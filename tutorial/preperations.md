# Preparations

## What this tutorial covers

This tutorial walks through the RIME-X workflow end to end, using a small example indicator (heating-degree-days) so everything runs quickly on a laptop with the bundled example data:

1. `001_preprocessing_data_setup.ipynb` -- defining indicators and telling RIME-X where their source data lives (download from ISIMIP, derive from other indicators, or register local data)
2. `002_preprocessing_regional_averages.ipynb` -- aggregating gridded indicator data to regions using area/population/etc. weighted masks
3. `003_preprocessing_quantilemaps.ipynb` -- building quantile maps that summarize the indicator's distribution across models and scenarios, by global warming level
4. `004_emulations.ipynb` -- combining quantile maps with a GMT ensemble (from a simple climate model) to emulate the indicator's distribution over time, for a given scenario
5. `005_run_FaIR.ipynb` -- generating your own GMT ensemble with FaIR, if you don't already have one from another SCM

By the end, you'll have a working pipeline from raw climate model output to emulated indicator timeseries under a scenario of your choice.

## Setting up the environment

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/iiasa/rimeX.git
cd rimeX
pip install -e .[all]
```

The `[all]` extra pulls in the optional dependencies needed across the tutorial (e.g. `xarray`, `jupyter`). If you plan to run the preprocessing steps yourself on your own raw ISIMIP data (rather than just the provided example files), you'll also need **CDO**:

```bash
# Linux
sudo apt-get install cdo

# macOS
brew install cdo

# or via conda (works everywhere)
conda install -c conda-forge cdo
```

### Using conda instead

```bash
conda create -n rimex-env python=3.10
conda activate rimex-env
conda install -c conda-forge cdo
cd rimeX
pip install -e .[all]
```

## Setting up the Jupyter kernel

The tutorial notebooks need to run with a kernel that has `rimeX` and its dependencies installed -- i.e. the same environment you just created above, not whatever default kernel Jupyter happens to start with.

1. With your `rimeX` environment activated, install `ipykernel` if it isn't already present:

   ```bash
   pip install ipykernel
   ```

2. Register the environment as a Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name=rimex --display-name="rimeX"
   ```

   (If you're using the conda environment above, run this from inside `rimex-env` instead.)

3. Launch Jupyter and open any of the tutorial notebooks:

   ```bash
   jupyter notebook
   ```

4. In the notebook, select the kernel via **Kernel > Change Kernel > rimeX** (or pick it from the kernel dropdown when the notebook first opens). If a notebook was previously opened with a different kernel, double-check this selection before running any cells -- an unrelated kernel will be missing the `rimeX` package entirely and every import will fail.

Once the `rimeX` kernel is selected, you're ready to start with `001_preprocessing_data_setup.ipynb`.