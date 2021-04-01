import os
import tarfile
import zipfile
from pathlib import Path

import toml


def get_file_name(url):
    """

    :param url: file url
    :return: file_name, file_ext, odir
    """
    base_name = os.path.basename(url)
    base_name = str.split(base_name, sep="?")[0]
    file_name, file_ext = (
        base_name.rsplit(".", 1))

    return file_name, file_ext


f_name, f_ext = get_file_name(
    "../locationdetection_acoustics/locationdetection_acoustics/input_dir/TAU-urban-acoustic-scenes-2019-development.audio.1.zip")

print(f"f_name is : {f_name} f_ext is {f_ext}")
