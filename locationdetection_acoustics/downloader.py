from pathlib import Path, PurePath
import toml
import zipfile
import tarfile
import os


def get_file_name(url):
    """

    :param url: file url
    :return: file_name, file_ext, odir
    """
    base_name = PurePath(url).name
    base_name = str.split(base_name, sep="?")[0]
    file_name, file_ext = base_name.rsplit(".", 1)

    return str(file_name), str(file_ext)


def file_extraction(file_tmp, file_ext, odir):
    """

    :param file_tmp: the downloaded file
    :param file_ext: the extension of the file
    :param odir: the output directory of the file
    :return:
    """
    odir = os.path.abspath(odir)
    odir.replace(".", "_")
    print(f" Extracting file to : {odir}")
    if not os.path.isdir(odir):
        os.makedirs(odir)
        if file_ext == "zip":
            with zipfile.ZipFile(file_tmp) as zip_ref:
                zip_ref.extractall(odir)
                os.remove(os.path.abspath(file_tmp))

        if file_ext == "tar" or file_ext == "tar.gz":
            tar = tarfile.open(file_tmp)
            tar.extractall(odir)


def data_download(datadict, file_path):
    """

    :param datadict:
    :param file_path:
    :return:
    """
    down_path = Path(file_path)
    if not down_path.is_dir():
        down_path.mkdir(parents=True)
    for _, url_dat in enumerate(datadict):
        url = datadict[url_dat][0]
        file_name, file_ext = get_file_name(url)
        odir = datadict[url_dat][1] + "/" + file_name
        file_down_path = down_path / (file_name + "." + file_ext)
        extracted_fold = Path((str(file_down_path)).rsplit(".", 1)[0])
        if not (file_down_path.exists() or extracted_fold.is_dir()):
            os.system(f"wget {url} -O {file_down_path}")
            file_extraction(file_down_path, file_ext, odir)


def download_process():
    # removed locationdetection_acoustics/ from the beginning
    config = toml.load(Path(r"../locationdetection_acoustics/locationdetection_acoustics/download.toml"))
    file_path = config["download_path"]
    down_paths = config["datadict"]
    data_download(down_paths, file_path)

