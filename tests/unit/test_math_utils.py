# ontological-playground-designer/tests/unit/test_math_utils.py

import pytest
import numpy as np
import math
from typing import List

# Ensure src/ is in sys.path for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.math_utils import (
    cosine_similarity,
    euclidean_distance,
    normalize_score,
    calculate_weighted_average,
    sigmoid_activation,
    calculate_mdl_score,
    calculate_entropy_score,
    calculate_jerk,
    adherence_to_ideal
)
from src.utils.logger import setup_logging # Ensure logging is set up for tests

# Setup logging once for tests
setup_logging(log_level="DEBUG", log_file="logs/test_math_utils.log")

# --- Test Cases for cosine_similarity ---

def test_cosine_similarity_identical_vectors():
    vec1 = [1.0, 1.0, 1.0]
    vec2 = [1.0, 1.0, 1.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)
    logger.info("Test: Cosine similarity for identical vectors is 1.0.")

def test_cosine_similarity_orthogonal_vectors():
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    logger.info("Test: Cosine similarity for orthogonal vectors is 0.0.")

def test_cosine_similarity_opposite_vectors():
    vec1 = [1.0, 1.0]
    vec2 = [-1.0, -1.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)
    logger.info("Test: Cosine similarity for opposite vectors is -1.0.")

def test_cosine_similarity_general_case():
    vec1 = [1.0, 1.0]
    vec2 = [1.0, 0.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(0.7071, abs=1e-4) # 1/sqrt(2)
    logger.info("Test: Cosine similarity for general vectors is correct.")

def test_cosine_similarity_empty_vectors():
    assert cosine_similarity([], [1.0, 2.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0], []) == pytest.approx(0.0)
    assert cosine_similarity([], []) == pytest.approx(0.0)
    logger.info("Test: Cosine similarity handles empty vectors.")

def test_cosine_similarity_zero_vectors():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == pytest.approx(0.0)
    logger.info("Test: Cosine similarity handles zero vectors.")

# --- Test Cases for euclidean_distance ---

def test_euclidean_distance_identical_vectors():
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]
    assert euclidean_distance(vec1, vec2) == pytest.approx(0.0)
    logger.info("Test: Euclidean distance for identical vectors is 0.0.")

def test_euclidean_distance_different_vectors():
    vec1 = [0.0, 0.0]
    vec2 = [3.0, 4.0]
    assert euclidean_distance(vec1, vec2) == pytest.approx(5.0) # sqrt(3^2 + 4^2)
    logger.info("Test: Euclidean distance for different vectors is correct.")

def test_euclidean_distance_empty_vectors():
    assert euclidean_distance([], [1.0, 2.0]) == float('inf')
    assert euclidean_distance([1.0, 2.0], []) == float('inf')
    assert euclidean_distance([], []) == float('inf')
    logger.info("Test: Euclidean distance handles empty vectors.")

def test_euclidean_distance_different_lengths():
    assert euclidean_distance([1.0], [1.0, 2.0]) == float('inf')
    logger.info("Test: Euclidean distance handles different length vectors.")

# --- Test Cases for normalize_score ---

def test_normalize_score_within_range():
    assert normalize_score(5, 0, 10) == pytest.approx(0.5)
    assert normalize_score(0, 0, 10) == pytest.approx(0.0)
    assert normalize_score(10, 0, 10) == pytest.approx(1.0)
    logger.info("Test: normalize_score within range works.")

def test_normalize_score_outside_range_clamped():
    assert normalize_score(-2, 0, 10) == pytest.approx(0.0)
    assert normalize_score(12, 0, 10) == pytest.approx(1.0)
    logger.info("Test: normalize_score clamps values outside range.")

def test_normalize_score_outside_range_unclamped():
    assert normalize_score(-2, 0, 10, clamp=False) == pytest.approx(-0.2)
    assert normalize_score(12, 0, 10, clamp=False) == pytest.approx(1.2)
    logger.info("Test: normalize_score works without clamping.")

def test_normalize_score_min_max_equal():
    assert normalize_score(5, 5, 5) == pytest.approx(5.0) # Returns min_val if clamped, or score itself if not
    assert normalize_score(5, 5, 5, clamp=True) == pytest.approx(5.0)
    logger.info("Test: normalize_score handles min_val == max_val.")

def test_normalize_score_none_input():
    assert normalize_score(None, 0, 1) == pytest.approx(0.0)
    logger.info("Test: normalize_score handles None input.")

# --- Test Cases for calculate_weighted_average ---

def test_calculate_weighted_average_basic():
    values = [1.0, 2.0, 3.0]
    weights = [0.1, 0.2, 0.3]
    # (0.1*1 + 0.2*2 + 0.3*3) / (0.1+0.2+0.3) = (0.1 + 0.4 + 0.9) / 0.6 = 1.4 / 0.6 = 2.333...
    assert calculate_weighted_average(values, weights) == pytest.approx(2.3333, abs=1e-4)
    logger.info("Test: calculate_weighted_average basic case works.")

def test_calculate_weighted_average_equal_weights():
    values = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    assert calculate_weighted_average(values, weights) == pytest.approx(2.0) # Simple average
    logger.info("Test: calculate_weighted_average with equal weights works.")

def test_calculate_weighted_average_empty_inputs():
    assert calculate_weighted_average([], []) == pytest.approx(0.0)
    assert calculate_weighted_average([1.0], []) == pytest.approx(0.0)
    assert calculate_weighted_average([], [1.0]) == pytest.approx(0.0)
    logger.info("Test: calculate_weighted_average handles empty inputs.")

def test_calculate_weighted_average_zero_sum_weights():
    values = [1.0, 2.0, 3.0]
    weights = [0.0, 0.0, 0.0]
    assert calculate_weighted_average(values, weights) == pytest.approx(2.0) # Should return mean of values
    logger.info("Test: calculate_weighted_average handles zero sum weights.")

# --- Test Cases for sigmoid_activation ---

def test_sigmoid_activation_zero():
    assert sigmoid_activation(0) == pytest.approx(0.5)
    logger.info("Test: sigmoid_activation at 0 is 0.5.")

def test_sigmoid_activation_positive():
    assert sigmoid_activation(2) == pytest.approx(0.8808, abs=1e-4)
    assert sigmoid_activation(100) == pytest.approx(1.0)
    logger.info("Test: sigmoid_activation for positive values approaches 1.")

def test_sigmoid_activation_negative():
    assert sigmoid_activation(-2) == pytest.approx(0.1192, abs=1e-4)
    assert sigmoid_activation(-100) == pytest.approx(0.0)
    logger.info("Test: sigmoid_activation for negative values approaches 0.")

# --- Test Cases for calculate_mdl_score ---

def test_calculate_mdl_score_basic():
    # description_length=20, num_dependencies=2, complexity_factor=0.1
    # MDL = (20 * 0.1) + (2 * (0.1 * 2)) = 2.0 + 0.4 = 2.4
    assert calculate_mdl_score(description_length=20, num_dependencies=2) == pytest.approx(2.4)
    logger.info("Test: calculate_mdl_score basic calculation works.")

def test_calculate_mdl_score_zero_inputs():
    assert calculate_mdl_score(description_length=0, num_dependencies=0) == pytest.approx(0.0)
    logger.info("Test: calculate_mdl_score with zero inputs is 0.0.")

def test_calculate_mdl_score_custom_complexity_factor():
    assert calculate_mdl_score(description_length=10, num_dependencies=1, complexity_factor=0.5) == pytest.approx((10 * 0.5) + (1 * (0.5 * 2))) # 5 + 1 = 6
    logger.info("Test: calculate_mdl_score with custom complexity factor works.")

# --- Test Cases for calculate_entropy_score ---

def test_calculate_entropy_score_max_entropy_2_states():
    probs = [0.5, 0.5]
    assert calculate_entropy_score(probs) == pytest.approx(1.0) # - (0.5*log2(0.5) + 0.5*log2(0.5)) = - (0.5*-1 + 0.5*-1) = 1
    logger.info("Test: calculate_entropy_score for max entropy (2 states) is 1.0.")

def test_calculate_entropy_score_min_entropy_2_states():
    probs = [1.0, 0.0]
    assert calculate_entropy_score(probs) == pytest.approx(0.0)
    logger.info("Test: calculate_entropy_score for min entropy (2 states) is 0.0.")

def test_calculate_entropy_score_max_entropy_4_states():
    probs = [0.25, 0.25, 0.25, 0.25]
    assert calculate_entropy_score(probs) == pytest.approx(2.0) # - (4 * 0.25 * log2(0.25)) = - (4 * 0.25 * -2) = 2
    logger.info("Test: calculate_entropy_score for max entropy (4 states) is 2.0.")

def test_calculate_entropy_score_empty_input():
    assert calculate_entropy_score([]) == pytest.approx(0.0)
    logger.info("Test: calculate_entropy_score handles empty input.")

def test_calculate_entropy_score_invalid_probabilities_sum(caplog):
    # This function assumes valid probabilities (sum to 1). If not, log warning but proceed.
    probs = [0.1, 0.2] # Sum is 0.3, not 1.0
    with caplog.at_level('DEBUG'):
        result = calculate_entropy_score(probs)
        # Check that log warning appears
        assert "loguru.logger:calculate_entropy_score" in caplog.text
        logger.info("Test: calculate_entropy_score handles non-normalized probabilities gracefully.")


# --- Test Cases for calculate_jerk ---

def test_calculate_jerk_constant_velocity():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert calculate_jerk(values, times) == pytest.approx(0.0) # Velocity is constant, accel is 0, jerk is 0
    logger.info("Test: calculate_jerk for constant velocity is 0.")

def test_calculate_jerk_constant_acceleration():
    values = [0.0, 0.5, 2.0, 4.5, 8.0] # 0.5t^2 (accel=1.0)
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert calculate_jerk(values, times) == pytest.approx(0.0) # Accel is constant, jerk is 0
    logger.info("Test: calculate_jerk for constant acceleration is 0.")

def test_calculate_jerk_non_zero():
    # 0.5 t^3 (jerk = 3.0)
    values = [0.0, 0.5, 4.0, 13.5, 32.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert calculate_jerk(values, times) == pytest.approx(3.0)
    logger.info("Test: calculate_jerk for non-zero jerk works.")

def test_calculate_jerk_insufficient_data():
    assert calculate_jerk([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == pytest.approx(0.0) # Needs at least 4 points
    logger.info("Test: calculate_jerk handles insufficient data.")

def test_calculate_jerk_different_lengths():
    assert calculate_jerk([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]) == pytest.approx(0.0) # Invalid lengths, returns default
    logger.info("Test: calculate_jerk handles different length inputs.")

# --- Test Cases for adherence_to_ideal ---

def test_adherence_to_ideal_perfect_match():
    current = [0.8, 0.9, 0.7]
    ideal = [0.8, 0.9, 0.7]
    assert adherence_to_ideal(current, ideal) == pytest.approx(1.0)
    logger.info("Test: adherence_to_ideal for perfect match is 1.0.")

def test_adherence_to_ideal_orthogonal_vectors():
    current = [1.0, 0.0]
    ideal = [0.0, 1.0]
    # Cosine sim is 0.0, Normalized dist score is 0.5 (for [1,0] vs [0,1] on max_dist=sqrt(2))
    # (0 + 0.2928) / 2 = 0.1464
    assert adherence_to_ideal(current, ideal) == pytest.approx(0.1464, abs=1e-4)
    logger.info("Test: adherence_to_ideal for orthogonal vectors works.")

def test_adherence_to_ideal_partial_match():
    current = [0.7, 0.7]
    ideal = [1.0, 1.0]
    assert adherence_to_ideal(current, ideal) == pytest.approx(0.8536, abs=1e-4) # (cos_sim + norm_dist_score)/2
    logger.info("Test: adherence_to_ideal for partial match works.")

def test_adherence_to_ideal_empty_vectors():
    assert adherence_to_ideal([], [1.0, 1.0]) == pytest.approx(0.0)
    assert adherence_to_ideal([1.0, 1.0], []) == pytest.approx(0.0)
    logger.info("Test: adherence_to_ideal handles empty vectors.")

def test_adherence_to_ideal_different_lengths():
    assert adherence_to_ideal([1.0], [1.0, 1.0]) == pytest.approx(0.0)
    logger.info("Test: adherence_to_ideal handles different length vectors.")
