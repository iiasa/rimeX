import pytest
import pandas as pd
import numpy as np

from rimeX.preproc.warminglevels import get_model_frequencies, get_matching_years_by_time_bucket

def test_get_model_frequencies():
    # Simulate a dataframe tracking when models hit warming levels
    df = pd.DataFrame({
        "model": ["ModelA", "ModelA", "ModelB"],
        "warming_level": [1.5, 2.0, 1.5]
    })
    
    freq = get_model_frequencies(df)
    
    # ModelA is at 1.5 and 2.0, ModelB is only at 1.5
    # For WL 1.5, there are 2 models. So freq should be 2.
    assert len(freq[1.5]) == 2
    
    # For WL 2.0, there is 1 model.
    assert len(freq[2.0]) == 1

def test_get_matching_years():
    # Mock year vs warming values for a model
    all_annual_df = pd.DataFrame({
        "tas": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        "year": [2010, 2011, 2012, 2013, 2014, 2015, 2016]
    }).set_index("year")
    
    # We want warming of 1.5
    warming_levels = [1.0, 1.5, 2.0]
    
    running_mean_window = 3 # 3-year bucket
    
    # The matching bucket around 2014 should be exactly 2013, 2014, 2015.
    result = get_matching_years_by_time_bucket(
        model="ModelA", 
        all_annual=all_annual_df, 
        warming_levels=warming_levels, 
        running_mean_window=running_mean_window, 
        projection_baseline=(1995, 2014)
    )
    
    # The logic appends multiple matching years
    assert len(result) == 5
    
    # Verify 2014 is inside for warming level 1.5
    assert any(r["year"] == 2014 and r["warming_level"] == 1.5 for r in result)

def test_get_matching_years_edge_clipping():
    # When window asks for values outside bounds, it shrinks.
    all_annual_df = pd.DataFrame({
        "tas": [1.1, 1.2, 1.3],
        "year": [2010, 2011, 2012]
    }).set_index("year")
    
    warming_levels = [1.0, 1.1, 1.5]
    
    result = get_matching_years_by_time_bucket(
        model="ModelA", 
        all_annual=all_annual_df, 
        warming_levels=warming_levels, 
        running_mean_window=3, # expects [2009, 2010, 2011]
        projection_baseline=(1995, 2014)
    )
    
    # Should not match anything because the 3-year smoothed value 1.2 is out of the 1.1 +- step/2 interval
    assert len(result) == 0
