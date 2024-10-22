import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def generate_samples_from_gmm(data: pd.DataFrame, n_samples: int = 1000, n_components: int = 3) -> np.ndarray:
    """
    Fits a Gaussian Mixture Model (GMM) to the provided data and generates new samples.

    This function fits a GMM with a specified number of components to the input data and
    generates a specified number of new samples from the fitted model.

    Args:
        data (pd.DataFrame): The input data to fit the GMM, typically a DataFrame where each
            row represents a data point and each column represents a feature.
        n_samples (int, optional): The number of samples to generate from the fitted GMM.
            Defaults to 1000.
        n_components (int, optional): The number of components (clusters) for the GMM.
            Defaults to 3.

    Returns:
        numpy.ndarray: A 2D array of generated samples, where each row is a sample and each
        column corresponds to a feature dimension.

    """

    # Fit Gaussian Mixture Model to the data
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)

    # Generate new samples
    samples, _ = gmm.sample(n_samples)
    return samples


def make_visualisation(data, generated_df, save_path):

    # Get feature names from the columns of the DataFrame
    feature_names = data.columns[:-1]  # Exclude the 'target' column

    # Define color maps for original and generated samples
    original_colors = ["red", "blue", "green"]
    generated_colors = ["orange", "cyan", "lime"]

    # Create subplots with different combinations of dimensions
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Comparison of Original and Generated Samples", fontsize=16)

    # List of axis combinations
    combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Plot each combination
    for idx, (i, j) in enumerate(combinations):
        row = idx // 2
        col = idx % 2

        # Plot generated data points
        for k in range(3):
            axs[row, col].scatter(
                generated_df[generated_df["target"] == k].iloc[:, i],
                generated_df[generated_df["target"] == k].iloc[:, j],
                color=generated_colors[k],
                label=f"Generated Class {k}" if idx == 0 else "",
                alpha=0.3,
            )

        # Plot original data points
        for k in range(3):
            axs[row, col].scatter(
                data[data["target"] == k].iloc[:, i],
                data[data["target"] == k].iloc[:, j],
                color=original_colors[k],
                label=f"Original Class {k}" if idx == 0 else "",
                alpha=0.6,
                edgecolor="black",
            )

        # Set labels using the feature names
        axs[row, col].set_xlabel(feature_names[i])
        axs[row, col].set_ylabel(feature_names[j])
        axs[row, col].grid(True)

    # Show legend only once
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    # save_path = os.path.join(data_dir, "data/data_generated.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
