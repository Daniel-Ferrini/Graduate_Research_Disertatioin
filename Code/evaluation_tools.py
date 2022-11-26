from model_evaluation import eval_wrapper, get_model_stats, identify_switchback
from cdf_loader import get_distance
from utils import load_wind
import pandas as pd
import numpy as np
import data_model
from matplotlib import rcParams
rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt


def plot_labels(
    data: pd.DataFrame([float]),
    model_predictions: pd.DataFrame([float]),
):
    """
    Generate label prediction comparison plots.

    :param data: DataFrame containing raw radial field data
    :param model_predictions: DataFrame containing model predictions
    :return: plot of label data
    """

    # plot same canvas data
    fig, ax = plt.subplots(figsize=(13, 7))
    ax2 = ax.twinx()
    ax.plot(data, c="r", label="Magnetic Field", alpha=0.8)
    ax2.plot(model_predictions["K-Means"] + 0.045, "|", label="K-Means", c="#1EA6FA")
    ax2.plot(model_predictions["DBSCAN"] + 0.03, "|", label="DBSCAN", c="#FAAD43")
    ax2.plot(
        model_predictions["Hierarchical"] + 0.015,
        "|",
        label="Hierarchical",
        c="#AFFA50",
    )
    ax2.plot(model_predictions["GM"], "|", label="Gaussian Mixture", c="#DA37FA")

    # define axis properties
    ax.set_ylim([data.min() * 1.15, data.max() * 1.25])
    ax.set_ylabel(r"Radial Magnetic Field Density ($nT$)", fontsize=14)
    ax2.set_ylim([-0.015, 1.2])
    ax2.set_yticks([])
    plt.title("Switchback classification results of unsupervised models", fontsize=18)
    plt.legend(loc=1, prop={'size': 12})
    plt.tight_layout()
    plt.show()


def plot_modes(data: pd.DataFrame([float])):
    """
    Generate statistical mode plot and comparison.

    :param data: DataFrame containing raw radial field data
    :param model_predictions: DataFrame containing model predictions
    :return: statistical mode plot
    """

    mode1 = data_model.stat_mode_generator(data=data, cadence=60)
    mode3 = data_model.stat_mode_generator(data=data, cadence=180)
    mode5 = data_model.stat_mode_generator(data=data, cadence=300)

    # plot same canvas data
    fig, ax = plt.subplots(figsize=(13, 7))
    ax2 = ax.twinx()
    lns1 = ax.plot(data, c="r", label="Magnetic Field", alpha=0.2)
    lns2 = ax2.plot(mode1, c="b", label="1 Hour Statistical Mode", alpha=0.4)
    lns3 = ax2.plot(mode3, c="y", label="3 Hour Statistical Mode", alpha=0.8)
    lns4 = ax2.plot(mode5, c="k", label="5 Hour Statistical Mode", alpha=1)

    # define axis properties
    ax.set_ylim([data.min() * 1.15, data.max() * 1.25])
    ax.set_ylabel(r"Radial Magnetic Field Density ($nT$)", fontsize=14)
    ax2.set_ylim([data.min() * 1.15, data.max() * 1.25])
    ax2.set_yticks([])
    plt.title("Statistical mode comparison plot", fontsize=18)
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=1, prop={'size': 12})
    plt.tight_layout()
    plt.show()


def plot_model_statistics(model_statistics: pd.DataFrame([float]), cadence: int):
    """
    Plot bar graph of model statistical comparison.

    :param cadence: window size (in hours) within which modes will be assessed
    :param model_statistics: DataFrame containing model predictions
    :return: model statistics bar graph
    """

    # plot same canvas data
    fig, ax = plt.subplots(figsize=(13, 10))
    x = np.arange(4)
    ax.bar(x - 0.3, model_statistics["K-Means"] * 100, color="#1EA6FA", width=0.2)
    ax.bar(x - 0.1, model_statistics["DBSCAN"] * 100, color="#FAAD43", width=0.2)
    ax.bar(x + 0.1, model_statistics["Hierarchical"] * 100, color="#AFFA50", width=0.2)
    ax.bar(x + 0.3, model_statistics["GM"] * 100, color="#DA37FA", width=0.2)

    # define axis properties
    ax.set_ylim([0, 100])
    ax.set_ylabel(r"Evaluation Scores (%)", fontsize=14)
    ax.set_xlabel(r"Scoring Criteria", fontsize=14)
    ax.set_xticks([0, 1, 2, 3], ["Accuracy", "Precision", "Recall", "F1-Score"])
    plt.title(
        "Unsupervised model performance evaluation metrics for a {} minute mode sample cadence".format(
            cadence
        )
        ,
        fontsize=18
    )
    ax.legend(labels=["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"], prop={'size': 12})
    plt.tight_layout()
    plt.show()


def plot_orbit(orbital_index: str, model_predictions: pd.DataFrame([float])):
    """
    Plot overlaid measurements/predictions for single orbit.
    for orbital info see - https://sppgway.jhuapl.edu/psp_data_releases

    :param orbital_index: string indicating the orbital count
    :param model_predictions: DataFrame containing model predictions
    :return: overlaid orbital pattern
    """

    # calculate orbital components
    def calculate_orbit(data: pd.Series([float])) -> tuple:
        """
        Calculate orbital parameters required for plot.

        :param data: data pertaining to orbital period
        :return: eccentricity and perihelion parameters
        """

        c = 1.690614874e-12
        m = 1.989e30 + 620
        r_s = 696340000
        t = len(data) * 60  # get orbital period
        a = (((t ** 2) * c) * m) ** (1 / 3)  # get semi-major axis
        e = 1 - (data.min().item() * r_s / a)  # get eccentricity

        return e * -0.75, a

    def scale_data(dist: pd.Series([float]), idx=pd.DataFrame([float])) -> tuple:
        n = 1440
        new_index = pd.RangeIndex(len(dist) * (n + 1))
        new_df = pd.Series(np.nan, index=new_index)
        ids = np.arange(len(dist)) * (n + 1)
        new_df.loc[ids] = dist.values
        interp = new_df.interpolate()

        day = dist.index.get_loc(idx.index[0], method="nearest")
        start_index = day * (n + 1)

        return interp, start_index

    # orbital period index
    index = {
        "first": ["2018-09-23", "2018-12-10"],
        "second": ["2019-02-28", "2019-05-01"],
        "third": ["2019-07-28", "2019-10-02"],
        "forth": ["2020-01-01", "2020-02-23"],
        "fifth": ["2020-05-13", "2020-06-22"],
        "sixth": ["2020-09-12", "2020-10-12"],
        "seventh": ["2021-01-02", "2021-01-29"],
        "eighth": ["2021-04-18", "2021-05-11"],
        "ninth": ["2021-07-31", "2021-08-24"],
        "tenth": ["2021-11-10", "2021-11-28"],
    }

    # get psp radius
    distance = get_distance()
    period = index[orbital_index]
    orbit = distance.loc[period[0] : period[1]]
    psp_dist, start = scale_data(dist=orbit, idx=model_predictions)

    # calculate wind components
    wind_vel = load_wind(data=model_predictions).rolling(720).mean()
    wind_vel.reset_index(drop=True, inplace=True)

    # format and align data
    predictions = model_predictions.reset_index(drop=True) * 35
    fill = pd.DataFrame(
        np.zeros((start, 4)), columns=["K-Means", "DBSCAN", "Hierarchical", "GM"]
    )
    predictions = pd.concat([predictions, fill]).reset_index(drop=True)
    predictions = predictions.shift(start).rolling(720).mean()
    pre = len(predictions) - len(wind_vel)
    overlay = pd.concat([psp_dist, predictions, wind_vel], axis=1).fillna(0)

    # calculate orbital components
    e, a = calculate_orbit(data=overlay[0])
    theta = np.linspace(0, 2 * np.pi, len(overlay[0]))
    r = (a * (1 - e ** 2)) / (1 + e * np.cos(theta))
    s = pd.Series(r * (10 ** (-10)), index=overlay[0].index)

    # extract variables
    k_means_ = s.add(overlay["K-Means"] * 0.15)
    dbscan_ = s.add(overlay["DBSCAN"] * 12.5)
    hierarchical_ = s.add(overlay["Hierarchical"] * 0.15)
    gm_ = s.add(overlay["GM"])
    vel = s.add(overlay["Vel Mag"].shift(pre).fillna(0) * 12)

    # plot same canvas data
    fig, ax = plt.subplots(figsize=(13, 10))
    plt.axis("off")
    plt.axes(projection="polar")
    plt.scatter(0.1, 0, s=200, c="y", label="Sun")
    plt.plot(theta, vel, label="Wind Velocity", c="gray")
    plt.plot(theta, k_means_, label="K-Means", c="#1EA6FA", alpha=0.55)
    plt.plot(theta, dbscan_, label="DBSCAN", c="#FAAD43", alpha=0.85)
    plt.plot(theta, hierarchical_, label="Hierarchical", c="#AFFA50", alpha=0.70)
    plt.plot(theta, gm_, label="Gaussian Mixture", c="#DA37FA")
    plt.plot(theta, s, c="k", linestyle=":", label="PSP Orbital Trajectory")

    # define axis properties
    # plt.title(
    #     "Switchback prediction comparison across {} encounter orbital trajectory".format(
    #         orbital_index
    #     )
    # )
    plt.yticks([])
    plt.xticks([])
    plt.axis("off")
    plt.legend(loc=1, prop={'size': 12})
    plt.tight_layout()
    plt.show()


def plot_eval(data: pd.DataFrame([float]), model_predictions: pd.DataFrame([float])):
    """
    Generate evaluation plot and comparison.

    :param data: DataFrame containing raw radial field data
    :param model_predictions: DataFrame containing model predictions
    :return: statistical mode plot
    """

    def stats(
        stat_mode: pd.DataFrame([float]),
        model_pred: pd.DataFrame([float]) = model_predictions,
    ):
        """
        Calculate stats of model evaluation.

        :param stat_mode: statistical modes for particular cadence
        :param model_pred:
        :return:
        """
        stat_k_means = get_model_stats(
            stat_mode=stat_mode, model_predictions=model_pred["K-Means"]
        )
        stat_dbscan = get_model_stats(
            stat_mode=stat_mode, model_predictions=model_pred["DBSCAN"]
        )
        stat_hierarchical = get_model_stats(
            stat_mode=stat_mode, model_predictions=model_pred["Hierarchical"]
        )
        stat_gm = get_model_stats(
            stat_mode=stat_mode, model_predictions=model_pred["GM"]
        )

        results = pd.concat(
            [stat_k_means, stat_dbscan, stat_hierarchical, stat_gm], axis=1
        )
        results.columns = ["K-Means", "DBSCAN", "Hierarchical", "GM"]

        return results

    mode1 = identify_switchback(data=data, cadence=60)
    mode3 = identify_switchback(data=data, cadence=180)
    mode5 = identify_switchback(data=data, cadence=300)
    x = np.arange(4)

    # plot same canvas data
    fig, (ax0, ax1, ax2) = plt.subplots(
        nrows=3, constrained_layout=True, figsize=(13, 17)
    )
    mode_eval_1 = stats(stat_mode=mode1)
    ax0.bar(x - 0.3, mode_eval_1["K-Means"] * 100, color="#1EA6FA", width=0.2)
    ax0.bar(x - 0.1, mode_eval_1["DBSCAN"] * 100, color="#FAAD43", width=0.2)
    ax0.bar(x + 0.1, mode_eval_1["Hierarchical"] * 100, color="#AFFA50", width=0.2)
    ax0.bar(x + 0.3, mode_eval_1["GM"] * 100, color="#DA37FA", width=0.2)
    ax0.set_title("Cross-model evaluation with 60 minute mode cadence", fontsize=16)
    ax0.set_ylim([0, 100])
    ax0.set_ylabel(r"Evaluation Scores (%)", fontsize=16)
    ax0.set_xticks([0, 1, 2, 3], ["Accuracy", "Precision", "Recall", "F1-Score"], fontsize=16)

    mode_eval_2 = stats(stat_mode=mode3)
    ax1.bar(x - 0.3, mode_eval_2["K-Means"] * 100, color="#1EA6FA", width=0.2)
    ax1.bar(x - 0.1, mode_eval_2["DBSCAN"] * 100, color="#FAAD43", width=0.2)
    ax1.bar(x + 0.1, mode_eval_2["Hierarchical"] * 100, color="#AFFA50", width=0.2)
    ax1.bar(x + 0.3, mode_eval_2["GM"] * 100, color="#DA37FA", width=0.2)
    ax1.set_title("Cross-model evaluation with 180 minute mode cadence", fontsize=16)
    ax1.set_ylim([0, 100])
    ax1.set_ylabel(r"Evaluation Scores (%)", fontsize=16)
    ax1.set_xticks([0, 1, 2, 3], ["Accuracy", "Precision", "Recall", "F1-Score"], fontsize=16)

    mode_eval_3 = stats(stat_mode=mode5)
    ax2.bar(x - 0.3, mode_eval_3["K-Means"] * 100, color="#1EA6FA", width=0.2)
    ax2.bar(x - 0.1, mode_eval_3["DBSCAN"] * 100, color="#FAAD43", width=0.2)
    ax2.bar(x + 0.1, mode_eval_3["Hierarchical"] * 100, color="#AFFA50", width=0.2)
    ax2.bar(x + 0.3, mode_eval_3["GM"] * 100, color="#DA37FA", width=0.2)
    ax2.set_ylim([0, 100])
    ax2.set_ylabel(r"Evaluation Scores (%)", fontsize=16)
    ax2.set_title("Cross-model evaluation with 300 minute mode cadence", fontsize=16)
    ax2.set_xlabel(r"Scoring Criteria", fontsize=16)
    ax2.set_xticks([0, 1, 2, 3], ["Accuracy", "Precision", "Recall", "F1-Score"], fontsize=16)

    # define axis properties
    # plt.title("Cross-model evaluation for varying mode cadences", fontsize=16)
    plt.tight_layout()
    plt.legend(loc=1, prop={'size': 14})
    plt.show()

    print("____________________________________________")
    print("1 Hour:")
    print(mode_eval_1)
    print()
    print("3 Hour:")
    print(mode_eval_2)
    print()
    print("5 Hour:")
    print(mode_eval_3)
    print("____________________________________________")
    

if __name__ == "__main__":
    cadence = 300
    data, model_predictions, model_statistics = eval_wrapper(cadence=cadence)
    plot_labels(data=data, model_predictions=model_predictions)
    plot_modes(data=data)
    # plot_model_statistics(model_statistics=model_statistics, cadence=cadence)
    plot_orbit(orbital_index="forth", model_predictions=model_predictions)
    plot_eval(data=data, model_predictions=model_predictions)
