import matplotlib.pyplot as plt

def plot_stress_curve(strain, stress, stress_lower=None, stress_upper=None,
                      title=None, label="Prediction", ci=False):
    """
    Plot stress-strain curve with optional credible interval.

    Parameters
    ----------
    strain : array-like
        Strain values.
    stress : array-like
        Stress values (e.g., predicted mean or median).
    stress_lower : array-like, optional
        Lower bound of credible interval.
    stress_upper : array-like, optional
        Upper bound of credible interval.
    title : str
        Title of the plot.
    label : str
        Label for the curve.
    ci : bool
        Whether to show credible interval and legend (Meta only).
    """
    plt.figure(figsize=(7, 5))
    plt.plot(strain, stress, label=label if ci else None, color="blue")

    if ci and stress_lower is not None and stress_upper is not None:
        plt.fill_between(
            strain, stress_lower, stress_upper,
            alpha=0.5, label="MC Dropout Uncertainty", color="pink"
        )

    plt.xlabel("Strain")
    plt.ylabel("Stress (MPa)")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.5)

    if ci:
        plt.legend()

    plt.tight_layout()
    plt.show()
