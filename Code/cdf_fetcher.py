import requests
from bs4 import BeautifulSoup


def url_generator(
    start_year: int,
    start_month: int,
    start_file: int,
    time_span: int,
    solar_wind: bool,
) -> list:
    """
    Fetches respective urls for .cdf files over specified range.

    :param start_year: starting year which data range is to be taken (2018, 2019, 2020, 2021)
    :param start_month: starting month which data range is to be taken
    :param start_file: first file which data range is to be taken
    :param time_span: approximate number of days over which desired data is to be taken
    :param solar_wind: indicate whether to use solar wind data samples or not (default set to false)
    :return: list of urls corresponding to provided input parameters
    """

    # select verbose or non-verbose data
    if solar_wind:
        seed_url = "http://sprg.ssl.berkeley.edu/data/spp/pub/sci/sweap/spc/L3/"
    else:
        seed_url = (
            "http://research.ssl.berkeley.edu/data/psp/data/sci/fields/l2/mag_RTN_1min/"
        )

    # insert starting year
    url_list = []
    for i in range(3):
        url_list.append(seed_url + str(start_year + i) + "/")

    # reference url
    ref = url_list[0] + "/0{}/".format(start_month)[-3:]

    # get file content list
    url_list = list_file_directories(seed_url=url_list, solar_wind=solar_wind)

    # index starting month
    start_list = list(filter(lambda x: ref in x, url_list))[0]
    index = url_list.index(start_list)

    # fetch specified url instances
    url_list = url_list[index:][(start_file - 1) : (time_span + start_file - 1)]

    return url_list


def list_file_directories(seed_url: list, solar_wind: bool) -> list:
    """
    Generate list of all available files in specified directory.

    :param seed_url: url link to file directory
    :param solar_wind: indicate whether to use solar wind data samples or not (default set to false)
    :return: list of all files in url directory
    """

    # parse url data
    url_list = []
    for url in seed_url:
        for month in range(1, 13):
            page = requests.get(url + "/0{}/".format(month)[-3:]).text
            soup = BeautifulSoup(page, "html.parser")

            # append urls to list
            if solar_wind:
                file_list = [
                    url + "/0{}/".format(month)[-3:] + node.get("href")
                    for node in soup.find_all("a")
                    if node.get("href").endswith("_v01.cdf")
                ]
            else:
                file_list = [
                    url + "/0{}/".format(month)[-3:] + node.get("href")
                    for node in soup.find_all("a")
                    if node.get("href").endswith("_v02.cdf")
                ]
            url_list += file_list

    return url_list


def cdf_downloader(url: str, solar_wind: bool):
    """
    Downloads and overwrites .cdf file from provided url.

    :param url: web link to .cdf file
    :param solar_wind: indicate whether to use solar wind data samples or not (default set to false)
    :return: loaded .cdf file
    """

    r = requests.get(url, allow_redirects=True)
    if solar_wind:
        open("data_files/parker_wind_data.cdf", "wb").write(r.content)
    else:
        open("data_files/parker_mag_data.cdf", "wb").write(r.content)
