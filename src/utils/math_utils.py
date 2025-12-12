# ontological-playground-designer/src/utils/math_utils.py

import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Dict, Union, Callable
import math

# Ensure loguru is set up for structured logging
from loguru import logger
from src.utils.logger import setup_logging

# Setup logging for this module
setup_logging()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates the cosine similarity between two numerical vectors.
    Used for comparing semantic embeddings of axioms or rules.
    Returns 1.0 for identical vectors, 0.0 for orthogonal, -1.0 for opposite.
    """
    try:
        if not vec1 or not vec2:
            logger.warning("One or both vectors are empty. Returning 0.0 similarity.")
            return 0.0
        
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)
        
        if np.linalg.norm(np_vec1) == 0 or np.linalg.norm(np_vec2) == 0:
            logger.warning("One or both vectors are zero vectors. Returning 0.0 similarity.")
            return 0.0
            
        similarity = 1 - cosine(np_vec1, np_vec2)
        logger.debug(f"Calculated cosine similarity: {similarity:.4f}")
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates the Euclidean distance between two numerical vectors.
    Useful for measuring 'distance' in conceptual space.
    """
    try:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            logger.warning("Invalid vectors for Euclidean distance. Returning inf.")
            return float('inf')
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)
        distance = np.linalg.norm(np_vec1 - np_vec2)
        logger.debug(f"Calculated Euclidean distance: {distance:.4f}")
        return float(distance)
    except Exception as e:
        logger.error(f"Error calculating Euclidean distance: {e}")
        return float('inf')


def normalize_score(score: Union[float, int], min_val: float = 0.0, max_val: float = 1.0, clamp: bool = True) -> float:
    """
    Normalizes a score to a specified range (default 0 to 1).
    Optionally clamps the score within the range.
    """
    try:
        if score is None:
            return 0.0 # Or raise error, depending on desired behavior
        
        # Handle cases where min_val == max_val to prevent division by zero
        if max_val == min_val:
            return float(min_val) if clamp else float(score)

        normalized = (float(score) - min_val) / (max_val - min_val)
        if clamp:
            normalized = np.clip(normalized, 0.0, 1.0)
        logger.debug(f"Normalized score {score} to {normalized:.4f} within [{min_val}, {max_val}]")
        return float(normalized)
    except Exception as e:
        logger.error(f"Error normalizing score: {e}")
        return 0.0

def calculate_weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Calculates the weighted average of a list of values.
    Used for combining adherence scores or other metrics.
    """
    try:
        if not values or not weights or len(values) != len(weights):
            logger.warning("Invalid input for weighted average. Returning 0.0.")
            return 0.0
        
        np_values = np.array(values)
        np_weights = np.array(weights)
        
        sum_weights = np.sum(np_weights)
        if sum_weights == 0:
            logger.warning("Sum of weights is zero for weighted average. Returning mean of values.")
            return np.mean(np_values)
        
        weighted_avg = np.sum(np_values * np_weights) / sum_weights
        logger.debug(f"Calculated weighted average: {weighted_avg:.4f}")
        return float(weighted_avg)
    except Exception as e:
        logger.error(f"Error calculating weighted average: {e}")
        return 0.0

def sigmoid_activation(x: float) -> float:
    """Applies the sigmoid activation function (useful for probabilities or smooth transitions)."""
    try:
        result = 1 / (1 + math.exp(-x))
        logger.debug(f"Sigmoid({x:.2f}) = {result:.4f}")
        return result
    except OverflowError: # Handle very large or very small x
        return 0.0 if x < 0 else 1.0
    except Exception as e:
        logger.error(f"Error calculating sigmoid: {e}")
        return 0.5 # Default to 0.5 on error

def calculate_mdl_score(description_length: int, num_dependencies: int, complexity_factor: float = 0.1) -> float:
    """
    Calculates a proxy for Minimum Description Length (MDL) for a rule.
    Lower score indicates higher parsimony. This is a simplified proxy.
    """
    try:
        # Simplified: MDL increases with description length and number of dependencies
        # Complexity factor scales how much each element contributes.
        mdl = (description_length * complexity_factor) + (num_dependencies * (complexity_factor * 2))
        logger.debug(f"Calculated MDL score: {mdl:.4f}")
        return mdl
    except Exception as e:
        logger.error(f"Error calculating MDL score: {e}")
        return float('inf')

def calculate_entropy_score(probabilities: List[float]) -> float:
    """
    Calculates Shannon entropy for a list of probabilities.
    Higher entropy indicates more disorder/uncertainty.
    Assumes probabilities sum to 1.
    """
    try:
        np_probs = np.array(probabilities)
        # Filter out zero probabilities to avoid log(0)
        np_probs = np_probs[np_probs > 0]
        
        if len(np_probs) == 0:
            return 0.0 # No elements, no entropy

        entropy = -np.sum(np_probs * np.log2(np_probs))
        logger.debug(f"Calculated entropy score: {entropy:.4f}")
        return float(entropy)
    except Exception as e:
        logger.error(f"Error calculating entropy score: {e}")
        return 0.0

def calculate_jerk(values_over_time: List[float], time_steps: List[float]) -> float:
    """
    Calculates the 'jerk' (third derivative) from a series of values over time.
    Requires at least 4 points (for 3rd derivative approx).
    Useful for predicting the acceleration of flourishing (as in EPON/PGM).
    """
    try:
        if len(values_over_time) < 4 or len(values_over_time) != len(time_steps):
            logger.warning("Not enough data points for jerk calculation. Returning 0.0.")
            return 0.0

        # Convert to numpy arrays
        np_values = np.array(values_over_time)
        np_time = np.array(time_steps)

        # Calculate first derivative (velocity)
        velocity = np.diff(np_values) / np.diff(np_time)
        # Calculate second derivative (acceleration)
        acceleration = np.diff(velocity) / np.diff(np_time[:-1])
        # Calculate third derivative (jerk)
        jerk = np.diff(acceleration) / np.diff(np_time[:-2])
        
        # Return average jerk if multiple values are produced, or the single value
        if len(jerk) > 0:
            return float(np.mean(jerk))
        logger.debug("Jerk calculation resulted in empty array. Returning 0.0.")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating jerk: {e}")
        return 0.0

def adherence_to_ideal(current_vec: List[float], ideal_vec: List[float]) -> float:
    """
    Calculates a score (0.0 to 1.0) representing adherence to an ideal vector.
    1.0 means perfect alignment. Combines Euclidean distance and Cosine Similarity.
    """
    try:
        if not current_vec or not ideal_vec or len(current_vec) != len(ideal_vec):
            logger.warning("Invalid vectors for adherence_to_ideal. Returning 0.0.")
            return 0.0
        
        # Semantic alignment (cosine similarity)
        sem_align = cosine_similarity(current_vec, ideal_vec)
        
        # Magnitude closeness (Euclidean distance, normalized)
        dist = euclidean_distance(current_vec, ideal_vec)
        
        # Assuming max possible distance is known or can be estimated
        # For simplicity, let's just use 1 - normalized_distance (max possible dist sqrt(2*dim*max_val^2))
        # This is a heuristic: (1 - normalized_dist) * sem_align
        max_possible_dist = np.sqrt(len(current_vec) * (1.0**2 + 1.0**2)) # Assuming vectors are normalized 0-1
        normalized_dist_score = 1.0 - (dist / max_possible_dist) if max_possible_dist > 0 else 0.0
        
        # Combine: prioritize semantic alignment, then magnitude.
        # A simple product or weighted sum can work.
        adherence_score = (sem_align + normalized_dist_score) / 2.0
        
        adherence_score = np.clip(adherence_score, 0.0, 1.0) # Ensure score is within 0-1 range
        logger.debug(f"Calculated adherence to ideal: {adherence_score:.4f}")
        return float(adherence_score)
    except Exception as e:
        logger.error(f"Error in adherence_to_ideal: {e}")
        return 0.0


# --- Example Usage (for testing and demonstration) ---
if __name__ == "__main__":
    # Ensure src/utils directory and logger.py exist for setup_logging
    import os
    if not os.path.exists("src/utils"):
        os.makedirs("src/utils")
        # Assuming logger.py is already there or will be created next

    logger.info("--- Demonstrating Math Utilities ---")

    # Cosine Similarity
    vec1_emb = [0.1, 0.2, 0.3, 0.4]
    vec2_emb = [0.1, 0.2, 0.3, 0.4] # Identical
    vec3_emb = [0.5, -0.5, 0.1, -0.1] # Different
    logger.info(f"Cosine Similarity (identical): {cosine_similarity(vec1_emb, vec2_emb):.4f}")
    logger.info(f"Cosine Similarity (different): {cosine_similarity(vec1_emb, vec3_emb):.4f}")

    # Euclidean Distance
    logger.info(f"Euclidean Distance (identical): {euclidean_distance(vec1_emb, vec2_emb):.4f}")
    logger.info(f"Euclidean Distance (different): {euclidean_distance(vec1_emb, vec3_emb):.4f}")

    # Normalize Score
    logger.info(f"Normalized score (0-10 to 0-1): {normalize_score(5, 0, 10):.4f}")
    logger.info(f"Normalized score (over max, clamped): {normalize_score(12, 0, 10):.4f}")

    # Weighted Average
    values = [0.8, 0.9, 0.5]
    weights = [0.3, 0.6, 0.1]
    logger.info(f"Weighted Average: {calculate_weighted_average(values, weights):.4f}")

    # Sigmoid Activation
    logger.info(f"Sigmoid(0): {sigmoid_activation(0):.4f}")
    logger.info(f"Sigmoid(2): {sigmoid_activation(2):.4f}")
    logger.info(f"Sigmoid(-2): {sigmoid_activation(-2):.4f}")

    # Calculate MDL Score (proxy)
    logger.info(f"MDL Score (short desc, few deps): {calculate_mdl_score(description_length=20, num_dependencies=2):.4f}")
    logger.info(f"MDL Score (long desc, many deps): {calculate_mdl_score(description_length=100, num_dependencies=10):.4f}")

    # Calculate Entropy Score
    probs1 = [0.5, 0.5] # Max entropy for 2 states
    probs2 = [1.0, 0.0] # Min entropy
    probs3 = [0.25, 0.25, 0.25, 0.25] # Max entropy for 4 states
    logger.info(f"Entropy for [0.5, 0.5]: {calculate_entropy_score(probs1):.4f}")
    logger.info(f"Entropy for [1.0, 0.0]: {calculate_entropy_score(probs2):.4f}")
    logger.info(f"Entropy for [0.25, 0.25, 0.25, 0.25]: {calculate_entropy_score(probs3):.4f}")

    # Calculate Jerk
    values_t = [0.0, 0.1, 0.4, 0.9, 1.6, 2.5] # quadratic-like values
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    logger.info(f"Jerk of quadratic-like values: {calculate_jerk(values_t, times):.4f}")
    
    # Adherence to Ideal
    ideal_v = [0.8, 0.8, 0.8]
    current_v = [0.75, 0.8, 0.85]
    logger.info(f"Adherence to ideal (close): {adherence_to_ideal(current_v, ideal_v):.4f}")
    current_v_far = [0.1, 0.1, 0.1]
    logger.info(f"Adherence to ideal (far): {adherence_to_ideal(current_v_far, ideal_v):.4f}")
