import pandas as pd
import numpy as np
import cdf_fetcher
import cdf_loader
import pickle
import utils


def time_series_generator(
    start_year: int,
    start_month: int,
    start_file: int,
    time_span: int,
    solar_wind: bool,
) -> pd.DataFrame([float]):
    """
    Generates time series mag field data points over specified sample window.

    :param start_year: starting year which data range is to be taken (2018, 2019, 2020, 2021)
    :param start_month: starting month which data range is to be taken
    :param start_file: first file which data range is to be taken
    :param time_span: approximate number of days over which desired data is to be taken
    :param solar_wind: indicate whether to use solar wind data samples or not (default set to false)
    :return: dataframe containing mag field time series vector
    """

    # generate url list of data files spanning desired time frame
    url_list = cdf_fetcher.url_generator(
        start_year=start_year,
        start_month=start_month,
        start_file=start_file,
        time_span=time_span,
        solar_wind=solar_wind,
    )

    # load required data from url and overwrite file.
    if solar_wind:
        solar_data = pd.DataFrame(
            data=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            columns=["Vp_R", "Vp_T", "Vp_N", "mu_R", "mu_T", "mu_N"],
        )
    else:
        mag_data = pd.DataFrame(data=[[0.0, 0.0, 0.0]], columns=["B_R", "B_T", "B_N"])

    for url in url_list:
        cdf_fetcher.cdf_downloader(url=url, solar_wind=solar_wind)
        if solar_wind:
            partial_data = cdf_loader.load_solar_cdf(
                file_name="data_files/parker_wind_data.cdf"
            )
            solar_data = pd.concat([solar_data, partial_data])
        else:
            partial_data = cdf_loader.load_cdf(
                file_name="data_files/parker_mag_data.cdf"
            )
            mag_data = pd.concat([mag_data, partial_data])

    if solar_wind:
        solar_data[1:].to_pickle(path="data_files/solar_data")
    else:
        mag_data[1:].to_pickle(path="data_files/raw_data")

    return solar_data[1:] if solar_wind else mag_data[1:]


def impute_loaded_data() -> pd.DataFrame:
    """
    Load and clean raw magnetic field data.

    :return: cleaned magnetic field DataFrame
    """

    raw_data = pickle.load(open("data_files/raw_data", "rb"))
    clean_data = raw_data.dropna()

    return clean_data


def generate_features() -> pd.DataFrame([float]):
    """
    Generate feature set from cleaned data.

    :return: DataFrame containing model features
    """

    # load and impute raw data
    clean_data = impute_loaded_data()

    # get feature components
    theta = utils.get_theta(mag_data=clean_data)
    phi = utils.get_phi(mag_data=clean_data)
    diff_features = utils.get_diff_params(mag_data=clean_data)

    # compose feature array
    features = pd.concat([theta, phi, diff_features], axis=1).dropna()

    return features


def stat_mode_generator(data: pd.Series([float]), cadence: int) -> pd.Series([float]):
    """
    Generate statistical modes for model analysis and evaluation.

    :param data: radial magnetic field feature data from which modal analysis is based
    :param cadence: window size (in hours) within which modes will be assessed
    :param bins: number of bins from which a single cadence is to be subdivided
    :return: Series containing mode data extracted from features
    """

    def mode(data_window) -> float:
        count, division = np.histogram(data_window, bins=int(cadence / 6))

        # compute mean value of modal bin
        mode_index = np.argmax(count)
        bin_center = (division[mode_index] + division[mode_index + 1]) / 2

        return bin_center

    # generate rolling bins
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=cadence)
    modes = data.rolling(window=indexer).apply(lambda x: mode(x))
    modes.name = "Statistical Mode"

    # shift and step through mode sequence
    aligned_mode = modes.shift(periods=int(cadence / 2))[::cadence]

    return aligned_mode


if __name__ == "__main__":
    mag_df = time_series_generator(
        start_year=2019, start_month=8, start_file=9, time_span=45, solar_wind=False
    )
    # wind_df = time_series_generator(
    #     start_year=2020, start_month=1, start_file=8, time_span=45, solar_wind=True
    # )

    cdf_loader.plot_fields(dataframe=mag_df)
