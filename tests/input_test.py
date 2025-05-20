import pytest
import warnings
import os
import numpy as np
from poromat.pipeline.generate_predictions import generate_prediction


# 1. INPUT VALIDATION TESTS

def test_porosity_negative_error():
    """Test that negative porosity raises ValueError"""
    with pytest.raises(ValueError, match="porosity can not be negative"):
        generate_prediction("meta", porosity=-1, T=300, rate=1000)

def test_porosity_above_range_warns():
    """Test porosity > 40 triggers a warning"""
    with pytest.warns(UserWarning, match="recommended porosity from 0 to 40"):
        generate_prediction("meta", porosity=50, T=300, rate=1000)

def test_temperature_out_of_range_warns():
    """Test T outside 20-400 triggers a warning"""
    with pytest.warns(UserWarning, match="recommended T from 20 to 400 Celsius degrees"):
        generate_prediction("meta", porosity=10, T=450, rate=1000)

def test_strainrate_zero_error():
    """Test zero strainrate raises ValueError"""
    with pytest.raises(ValueError, match="strainrate must be positive"):
        generate_prediction("meta", porosity=10, T=300, rate=0)

def test_strainrate_out_of_range_warns():
    """Test strainrate < 500 or > 4500 triggers a warning"""
    with pytest.warns(UserWarning, match="recommended strainrate from 500 to 4500"):
        generate_prediction("meta", porosity=10, T=300, rate=100)

# 2. PREDICTION + PLOT TESTS

@pytest.mark.parametrize("model_name", ["lightgbm", "interpolation", "meta"])
def test_model_prediction_plot(model_name):
    """Test all models can run with valid inputs and plot output"""
    strain, stress, lower, upper = generate_prediction(
        model_name=model_name,
        porosity=10.0,
        T=300,
        rate=1000,
        show_plot=True  # Plot is not checked, only that it runs
    )
    assert isinstance(strain, np.ndarray)
    assert isinstance(stress, np.ndarray)
    assert len(strain) > 0
    assert len(stress) > 0

# 3. PREDICTION + SAVE CSV TESTS

@pytest.mark.parametrize("model_name", ["lightgbm", "interpolation", "meta"])
def test_model_prediction_save_csv(tmp_path, model_name):
    """Test all models can save predictions to CSV without error"""
    _ = generate_prediction(
        model_name=model_name,
        porosity=10.0,
        T=300,
        rate=1000,
        save_csv=True,
        output_dir=str(tmp_path)  # use temporary directory
    )

    # Check if at least one csv file is created in temp folder
    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) > 0
