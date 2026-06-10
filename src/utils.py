import numpy as np
def calculate_weights(total_samples, samples_in_class, num_classes):
    """Calculates class weights for imbalanced datasets."""
    weights = []
    for count in samples_in_class:
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    return weights

def mean_std_calculator(dataframe):
    """Calculates mean and std on [0, 1] scaled pixels."""
    all_pixels = np.vstack(dataframe['pixels'].values)
    scaled_pixels = all_pixels / 255.0
    return float(np.mean(scaled_pixels)), float(np.std(scaled_pixels))