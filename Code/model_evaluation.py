from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import k_means, dbscan, hierarchical, gaussian_mixture
import pandas as pd
import numpy as np
import data_model


def identify_switchback(
    data: pd.DataFrame([float]), cadence: int
) -> pd.DataFrame([int]):
    """
    Use modal data to identify switchbacks in dataset.

    :param data: original radial magnetic field data
    :param cadence: window size (in hours) within which modes will be assessed
    :return: DataFrame containing 'true' switchback labels
    """

    # extract and identify modes
    modes = data_model.stat_mode_generator(data=data, cadence=cadence)

    # perform modal comparison
    diff = pd.concat([data, modes], axis=1).fillna(method="ffill").dropna()

    diff[np.sign(diff["B_R"]) == np.sign(diff["Statistical Mode"])] = 0
    diff[np.sign(diff["B_R"]) != np.sign(diff["Statistical Mode"])] = 1

    return diff["Statistical Mode"]


def get_model_stats(
    stat_mode: pd.DataFrame([float]),
    model_predictions: pd.DataFrame([float]),
) -> pd.DataFrame([float]):
    """
    Calculate model performance metrics against statistical modes.

    :param stat_mode: true switchback labels determined by modal analysis
    :param model_predictions: predicted switchback labels
    :return: model performance statistics (accuracy, precision, recall, f1-score)
    """

    # align prediction results
    df = pd.concat([stat_mode, model_predictions], axis=1).dropna()
    df.columns = ["Y-true", "Y-pred"]

    # calculate predictions statistics
    statistics = {
        "Accuracy": accuracy_score(y_true=df["Y-true"], y_pred=df["Y-pred"]),
        "Precision": precision_score(y_true=df["Y-true"], y_pred=df["Y-pred"]),
        "Recall": recall_score(y_true=df["Y-true"], y_pred=df["Y-pred"]),
        "F1-Score": f1_score(y_true=df["Y-true"], y_pred=df["Y-pred"]),
    }

    model_statistics = pd.DataFrame.from_dict(statistics, orient="index")

    return model_statistics


def eval_wrapper(cadence: int) -> tuple:
    """
    Perform modal evaluation on statistical modes.

    :param cadence: window size (in hours) within which modes will be assessed
    :return: DataFrames containing model predictions and  performance statistics respectively
    """

    i = 10000
    t = 30000

    # generate features and comparison data
    features = data_model.generate_features()
    data = data_model.impute_loaded_data()["B_R"]
    stat_mode = identify_switchback(data=data, cadence=cadence)

    # fit unsupervised models
    k_means_ = k_means(features=features, sample_size=0.2)
    dbscan_ = dbscan(features=features)
    hierarchical_ = hierarchical(features=features)
    gm_ = gaussian_mixture(features=features, sample_size=0.2)

    # calculate model statistics
    stat_k_means = get_model_stats(stat_mode=stat_mode, model_predictions=k_means_)
    stat_dbscan = get_model_stats(stat_mode=stat_mode, model_predictions=dbscan_)
    stat_hierarchical = get_model_stats(
        stat_mode=stat_mode, model_predictions=hierarchical_
    )
    stat_gm = get_model_stats(stat_mode=stat_mode, model_predictions=gm_)

    # accumulate predictions and results
    predictions = pd.concat([k_means_, dbscan_, hierarchical_, gm_], axis=1)
    results = pd.concat([stat_k_means, stat_dbscan, stat_hierarchical, stat_gm], axis=1)

    predictions.columns = ["K-Means", "DBSCAN", "Hierarchical", "GM"]
    results.columns = ["K-Means", "DBSCAN", "Hierarchical", "GM"]

    return data, predictions, results


if __name__ == "__main__":
    raw_data, model_predictions, model_stats = eval_wrapper(cadence=120)
