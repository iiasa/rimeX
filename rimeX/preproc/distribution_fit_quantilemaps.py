"""Quantile maps for impacts derived from a distribution fit
"""
from itertools import groupby
import tqdm
from pathlib import Path
import argparse

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import xarray as xa
from rimeX.logs import logger, log_parser, setup_logger

from rimeX.config import CONFIG, config_parser
from rimeX.logs import logger, log_parser
from rimeX.stats import fast_quantile, fast_weighted_quantile, equally_spaced_quantiles
from rimeX.datasets.download_isimip import Indicator, get_models, isimip_parser
from rimeX.preproc.warminglevels import get_warming_level_file, get_root_directory
from rimeX.preproc.digitize import transform_indicator
from rimeX.preproc.regional_average import get_all_regions


def get_filepath(
    name,
    season="annual",
    root_dir=None,
    return_period=None,
    regional=False,
    regional_weight="latWeight",
    **kw,
):
    """Get file path for 

    Args:
        name (str): Indicator name
        season (str, optional): seasonality. Defaults to "annual".
        root_dir (str, Path): where to save the output. Defaults to None.
        return_period (int, optional): Return period of the indicator. Defaults to None.
        regional (bool, optional): True if return periods are for regionally aggregated values. Defaults to False.
        regional_weight (str, optional): Aggregation type (Needed if regional=True). Defaults to "latWeight".

    Returns:
        _type_: _description_
    """    
    if root_dir is None:
        root_dir = get_root_directory(**kw)
    if regional:
        return (
            root_dir
            / "quantilemaps_regional_admin"
            / f"{name}-rt{return_period}"
            / f"{name}-rt{return_period}_{season}_admin_{regional_weight.lower()}.nc"
        )
    else:
        return (
            root_dir
            / "quantilemaps"
            / f"{name}-rt{return_period}"
            / f"{name}-rt{return_period}_{season}_quantilemaps.nc"
        )

def create_quantile_map(indicator: Indicator, 
                        regional: bool,
                        aggregation:str,
                        models:list, 
                        impact_models: list, 
                        model_weights:dict, 
                        quantiles: list, 
                        return_periods:list):
    """Create quantile map file (dataarray with dimensions quantiles, warming levels, 
    region (either a region name or a lat lon pair)) from the output files from distribution_fit script. 
    This quantile map file will be saved as the indicator {indicator}-r{return_period} and can be emulated 
    with rimeX
    Args:
        indicator (Indicator): The indicator
        regional (bool): If True wqe calculate the quantile map from the GEV fits on the regional aggregation, otherwise for the latitude and longitude values
        aggregation (str): aggregation type (e.g. area weighted latWeight)
        models (list): models to consider
        impact_models (list): impact models to consider or None, if None the code assumes that there are no impact models relevant for the indicator
        model_weights (dict): dictionary with the models as keys and weights as values. If None every model gets an equal weight
        quantiles (list): all quantiles recoded in the quantile map
        return_periods (list): all return periods to produce a quantile map for
    """    

    input_paths = []

    if model_weights is None:
         weights = None
    else: 
        weights = []
        
    for model in models: 

        if model_weights is not None: 
     
            weights.append(model_weights[model])

        if impact_models is not None: 
        
            for impact_model in impact_models:

                if regional:
                    input_paths.append(Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/GEV_fits/{model}_{impact_model}_{indicator.name}_annual_return_periods_non_stationary_GEV_{aggregation}_regional.nc'))

                else:
                
                    input_paths.append(Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/GEV_fits/{model}_{impact_model}_{indicator.name}_annual_return_periods_non_stationary_GEV.nc'))

        else: 
            
            if regional: 
                input_paths.append(Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/GEV_fits/{model}_{indicator.name}_annual_return_periods_non_stationary_GEV_{aggregation}_regional.nc'))

            else: 
                input_paths.append(Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/GEV_fits/{model}_{indicator.name}_annual_return_periods_non_stationary_GEV.nc'))


    out_list = []
    
    for return_period in return_periods:

        out_path = get_filepath(indicator.name,
            season="annual",
            root_dir=None,
            return_period=return_period,
            regional=regional,
            regional_weight=aggregation)
        
        logger.info(f'Making {str(out_path)}')
        
        model_datasets = []
        
        for path in input_paths: 
      
            with xa.open_dataset(path) as f:
                data = f.sel(return_period = return_period).load()
            
            model_datasets.append(data)

        out_quantiles = []
        # Combine along a new dimension called 'model'
        model_data = xa.concat(model_datasets,  dim="model")
        model_data["model"] = models
        for warming_level in model_data.warming_level:

            quantile_wl = fast_weighted_quantile(
                    model_data.sel(warming_level = warming_level).rx5day, np.array(quantiles), weights=weights, dim="model", skipna=True
                )
    
            out_quantiles.append(quantile_wl)
        
        
        out_ret = xa.concat(out_quantiles,  dim="warming_level")
        
        out_ret['warming_level'] = model_data.warming_level
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        out_ret.to_netcdf(out_path)
        
        logger.info(f'Finished {str(out_path)}')
    




def main():
    ALL_MODELS = get_models(simulation_round=CONFIG['isimip.simulation_round'])
    DEFAULT_OUTPUT_RETURN_PERIODS = [2,20,50,100]#[2,3,4,5,6,7,8,9,10,15,20,25] + [30 + i*10 for i in range(8)] + [150] + [200 + i*100 for i in range(9)]
    all_variables = ['rx5day']
    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser, isimip_parser])
    # parser.add_argument("-v", "--variable", nargs='+', default=[], choices=CONFIG["isimip.variables"])
    parser.add_argument("-i", "--indicator", nargs='+', default=[], choices=all_variables, help="includes additional, secondary indicator with specific monthly statistics")
    parser.add_argument("--overwrite", action='store_true')
    # parser.add_argument("--backends", nargs="+", choices=["csv", "netcdf"], default=["netcdf", "csv"])
    #parser.add_argument("--frequency")
    group = parser.add_argument_group('mask')
    #group.add_argument("--region", nargs='+', default=ALL_REGIONS, choices=ALL_REGIONS)
    #group.add_argument("--weights", nargs='+', default=CONFIG["preprocessing.regional.weights"], choices=CONFIG["preprocessing.regional.weights"])
    group.add_argument("--regional", action='store_true')
    #group.add_argument("--regions", nargs='+', help="Choose subset of all regions with available masks. If left empty every region will be considered by default. Only applies when regional flac is active.")
    group.add_argument("--aggregation", default ='latWeight', help="Aggregation type")
    #ägroup.add_argument("--cpus", type=int, default = 1)
    #group.add_argument("--warming_level_file", type=int)
    group.add_argument("--models", nargs='+',default = ALL_MODELS, help="All models considered in the processing, they have to have at least one simulation available for the indicator. If left empty every model providing simulations for the indicator will be considered by default.")
    group.add_argument("--impact_models", nargs='+',default = None, help="All impact models considered in the processing, they have to have at least one simulation available for the indicator. If left empty every impact model providing simulations for the indicator will be considered by default.")
    #group.add_argument("--warming_levels_output", nargs='+', default = DEFAULT_OUTPUT_WARMING_LEVELS, help="Warming levels for which return periods from non-stationary GEV fit should be recorded in output file.")
    group.add_argument("--return_periods_output", nargs='+', default = DEFAULT_OUTPUT_RETURN_PERIODS, help="Return periods from non-stationary GEV fit which should be recorded in output file.")

    o = parser.parse_args()
    setup_logger(o)

    for indicator_name in o.indicator:
        indicator = Indicator.from_config(indicator_name)

      
        create_quantile_map(indicator = indicator, 
                            regional = o.regional,
                            aggregation = o.aggregation,
                            models = o.models, 
                            impact_models = o.impact_models, 
                            model_weights = None, 
                            quantiles = [ 0.01*i for i in range(101)], 
                            return_periods = o.return_periods_output)

    



if __name__ == "__main__":
    main()
