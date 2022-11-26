import cdflib
import datetime
import warnings
import pandas as pd
from matplotlib import rcParams

rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
warnings.warn(
    "/Users/danielferrini/Documents/Masters Disertation/Code/cdf_loader.py:62: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()` dis_vals[elements[0]] = elements[1:]"
)


def convert_time(time_stamp: list) -> list:
    """
    Converts time stamp from J2000 to UTC format.

    time_stamp: list of J2000 time stamps
    :return: utc time format
    """

    # initialise time list
    time = []

    # convert time intervals to utc
    for instance in time_stamp:
        interval = datetime.datetime(2000, 1, 1, 12, 0) + datetime.timedelta(
            seconds=instance * (10 ** (-9))
        )
        time.append(interval)

    return time


def get_distance() -> pd.Series([float]):
    """
    Load distance and format distance instances from .dat file

    :return: Series of timestamped distance metrics
    """

    # initialise distance vales dataframe
    dis_vals = pd.DataFrame(
        [],
        columns=[
            "Epoch",
            "RAD_AU",
            "SE_LAT",
            "SE_LON",
            "HG_LAT",
            "HG_LON",
            "HGI_LAT",
            "HGI_LON",
        ],
        index=[0],
    ).T

    # load and tokenize .dat file
    file = "data_files/distance.dat"
    for line in open(file, "r"):
        item = line.rstrip()
        if len(item) == 94:
            elements = item.split()
            elements[1:3] = [" ".join(elements[1:3])]
            if len(elements) == 9:
                dis_vals[elements[0]] = elements[1:]

    dis_vals = dis_vals.T.dropna()

    # format data arrays
    dis_vals["Epoch"] = pd.to_datetime(dis_vals["Epoch"])
    dis_vals["RAD_AU"] = pd.to_numeric(dis_vals["RAD_AU"])

    # generate feature series
    psp_distance = pd.Series(dis_vals["RAD_AU"], name="Distance")
    psp_distance.index = dis_vals["Epoch"]

    return psp_distance


def index_dist(time: list, distance: pd.Series([float])) -> list:
    """
    Get closest distance value at specified time instance.

    :param time: time sample array for given distance
    :param distance: distance of psp at sampled time
    :return: interpolated distance array for psp
    """

    dist_array = []
    for instance in time:
        nearest_time = distance.index.get_loc(instance, method="nearest")
        nearest_dist = distance[nearest_time]
        dist_array.append(nearest_dist)

    return dist_array


def load_cdf(file_name: str) -> pd.DataFrame([float]):
    """
    Load specified .cdf file.

    :param file_name: directory name for desired .cdf
    :return: data and time stamp variables
    """

    # load .cdf file
    cdf_file = cdflib.CDF(file_name)

    # extract data and time variables
    data = cdf_file.varget(variable=1)
    sat_time = cdf_file.varget(variable=0)

    # convert time to utc
    time = convert_time(time_stamp=sat_time)
    distance = get_distance()
    psp_dist = index_dist(time=time, distance=distance)

    # map time variables to mag field data
    df = pd.DataFrame(data=data, columns=["B_R", "B_T", "B_N"], index=time)
    df["Distance"] = psp_dist

    return df


def load_solar_cdf(file_name: str) -> pd.DataFrame([float]):
    """
    Load specified solar wind .cdf file.

    :param file_name: directory name for desired .cdf
    :return: data and time stamp variables
    """

    # load .cdf file
    cdf_file = cdflib.CDF(file_name)

    # 'vp_fit_RTN', 'vp_fit_RTN_uncertainty' : 8, 9
    # 'vp1_fit_RTN', 'vp1_fit_RTN_uncertainty' : 16, 17
    # 'va_fit_RTN', 'va_fit_RTN_uncertainty' : 36, 37
    # 'v3_fit_RTN', 'v3_fit_RTN_uncertainty' : 44, 45

    # extract data and time variables
    data = cdf_file.varget(variable=8)
    uncertainty = cdf_file.varget(variable=9)
    sat_time = cdf_file.varget(variable=0)

    # convert time to utc
    time = convert_time(time_stamp=sat_time)

    vel_df = pd.DataFrame(
        data=data,
        columns=["Vp_R", "Vp_T", "Vp_N"],
        index=time,
    )
    unc_df = pd.DataFrame(
        data=uncertainty,
        columns=["mu_R", "mu_T", "mu_N"],
        index=time,
    )

    df = pd.concat([vel_df, unc_df], axis=1)

    return df


def plot_fields(dataframe: pd.DataFrame, mean_plot: bool = True):
    """
    Generate magnetic field plot from provided data.

    :param dataframe: dataframe containing RTN magnetic field data
    :param mean_plot: indicate whether to plot rolling average field
    :return: plot of RTN field measurements
    """

    fig, axs = plt.subplots(3, figsize=(13, 17))
    plt.suptitle("Parker Solar Probe RTN Magnetic Field Measurements\n", fontsize=18)

    # radial field plot
    axs[0].plot(dataframe["B_R"], color="r")
    axs[0].set_title("Radial Magnetic Field", fontsize=16)
    axs[0].set_ylabel(r"$B_R\:nT$", fontsize=16)

    # tangential field plot
    axs[1].plot(dataframe["B_T"], color="b")
    axs[1].set_title("\nTangential Magnetic Field", fontsize=16)
    axs[1].set_ylabel(r"$B_T\:nT$", fontsize=16)

    # normal field plot
    axs[2].plot(dataframe["B_N"], color="y")
    axs[2].set_title("\nNormal Magnetic Field", fontsize=16)
    axs[2].set_ylabel(r"$B_N\:nT$", fontsize=16)

    # rolling average plot
    if mean_plot:
        axs[0].plot(
            dataframe["B_R"].rolling(window=1000).mean()[:-1000],
            color="black",
            linestyle="dashed",
        )
        axs[1].plot(
            dataframe["B_T"].rolling(window=1000).mean()[:-1000],
            color="black",
            linestyle="dashed",
        )
        axs[2].plot(
            dataframe["B_N"].rolling(window=1000).mean()[:-1000],
            color="black",
            linestyle="dashed",
        )

    fig.tight_layout()
    plt.show()
