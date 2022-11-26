from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle


def load_wind(data: pd.DataFrame([float])) -> pd.DataFrame([float]):
    '''
    Load solar wind data and return radial velocity components and uncertainty.

    :param data: data containing index of comparison
    :return: tuple containing wind velocity and wind uncertainty
    '''

    # load solar wind data
    df = pickle.load(open("data_files/solar_data", 'rb'))
    df = df.set_index(pd.DatetimeIndex(df.index)).resample("1T").mean()
    df[df < -5e-30] = None
    df.fillna(method="bfill", inplace=True)

    # isolate relevant index
    data, df = data.reset_index(), df.reset_index()
    indexed_df = pd.merge_asof(data, df, on="index", direction="nearest")
    indexed_df.set_index("index", inplace=True)

    # get magnitude quantities
    vel_mag = (indexed_df["Vp_R"] ** 2 + indexed_df["Vp_T"] ** 2 + indexed_df["Vp_N"] ** 2) ** 0.5
    vel_mag.name = "Vel Mag"

    # scale all
    vel_mag = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min())

    return vel_mag


def get_theta(mag_data: pd.DataFrame([float])) -> pd.Series([float]):
    """
    Calculate theta parameter for every in-situ magnetic field measurement.

    :param mag_data: cleaned RTN magnetic field data.
    :return: Series of magnetic field theta component.
    """

    # get mag field components and calculate theta
    t_component, n_component = mag_data["B_T"], mag_data["B_N"]
    theta = np.arctan2(n_component, t_component)
    theta.name = "Theta"

    return theta


def get_phi(mag_data: pd.DataFrame([float])) -> pd.Series([float]):
    """
    Calculate phi parameter for every in-situ magnetic field measurement.

    :param mag_data: cleaned RTN magnetic field data.
    :return: Series of magnetic field phi component.
    """

    # get mag field components and calculate phi
    r_component, t_component = mag_data["B_R"], mag_data["B_T"]
    phi = np.arctan2(t_component, r_component)
    phi.name = "Phi"

    return phi


def get_diff_params(mag_data: pd.DataFrame([float])) -> pd.DataFrame([float]):
    """
    Calculate differential parameters for every in-situ magnetic field measurement.

    :param mag_data: cleaned RTN magnetic field data.
    :return: DataFrame of boolean and differential components.
    """

    # calculate sign and difference between radial measurements
    diff_params = pd.DataFrame(columns=["Pct Change", "Distance"])
    r_component, distance = mag_data["B_R"], mag_data["Distance"]
    diff_params["Pct Change"] = r_component.pct_change()
    diff_params["Distance"] = distance

    return diff_params


def std_scalar(data: pd.DataFrame([float])) -> pd.DataFrame([float]):
    """
    Perform Gaussian normalisation on feature data, and export scalar model.

    :param data: data to be standardised
    :return: normalised data array
    """

    # initialise and fit scalar model
    scalar = StandardScaler()
    scaled_data = scalar.fit_transform(data)

    # convert scaled data to DataFrame
    features = pd.DataFrame(data=scaled_data, columns=data.columns, index=data.index)

    return features


def random_subset(
    full_dataset: pd.DataFrame([float]), sample_size: float
) -> pd.DataFrame([float]):
    """
    Generate a random sample subset from the full dataset.

    :param full_dataset: dataset to be sampled
    :param sample_size: percentage of dataset to be randomly
    :return: Dataframe containing randomly sampled data subset
    """

    sample_set = full_dataset.sample(frac=sample_size, random_state=42)
    return sample_set


def k_means(features: pd.DataFrame([float]), sample_size=float) -> pd.Series([int]):
    """
    Perform binary k-means clustering on feature data.

    :param features: cleaned model feature array
    :param sample_size: percentage of dataset to be randomly
    :return: cluster labels and fitted k-means model
    """

    # define and fit k-means model
    normalised_features = std_scalar(data=features)
    sampled_features = random_subset(
        full_dataset=normalised_features, sample_size=sample_size
    )
    k_means_model = KMeans(n_clusters=2, random_state=42).fit(sampled_features)
    pickle.dump(k_means_model, open("model_files/k_means.pkl", "wb"))

    # generate label predictions
    labels = pd.Series(k_means_model.predict(features), index=features.index)

    return labels


def dbscan(features: pd.DataFrame([float])) -> pd.Series([int]):
    """
    Perform binary DBSCAN clustering on feature data.

    :param features: cleaned model feature array
    :return: cluster labels and fitted DBSCAN model
    """

    # define and fit dbscan model
    dbscan_model = DBSCAN(eps=4, min_samples=8).fit(features)
    pickle.dump(dbscan_model, open("model_files/dbscan.pkl", "wb"))

    # generate label predictions
    labels = pd.Series(dbscan_model.labels_ * -1, index=features.index)

    return labels


def hierarchical(features: pd.DataFrame([float])) -> pd.Series([int]):
    """
    Perform binary hierarchical clustering on feature data.

    :param features: cleaned model feature array
    :return: cluster labels and fitted hierarchical model
    """

    # define and fit hierarchical model
    normalised_features = std_scalar(data=features)
    hierarchical_model = AgglomerativeClustering(n_clusters=2).fit(normalised_features)
    pickle.dump(hierarchical_model, open("model_files/hierarchical.pkl", "wb"))

    # generate label predictions
    labels = pd.Series(hierarchical_model.labels_, index=features.index)

    return labels


def gaussian_mixture(
    features: pd.DataFrame([float]), sample_size=float
) -> pd.Series([int]):
    """
    Perform binary baysian gaussian mixture clustering on feature data.

    :param features: cleaned model feature array
    :param sample_size: percentage of dataset to be randomly
    :return: cluster labels and fitted gaussian mixture model
    """

    # define and fit BGM model
    sampled_features = random_subset(full_dataset=features, sample_size=sample_size)
    bgm_model = BayesianGaussianMixture(n_components=2, random_state=42).fit(
        sampled_features
    )
    pickle.dump(bgm_model, open("model_files/gaussian_mixture.pkl", "wb"))

    # generate label predictions
    labels = pd.Series(bgm_model.predict(features), index=features.index)

    return labels
