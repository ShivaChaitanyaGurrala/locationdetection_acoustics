from downloader import download_process
from preprocess import Preprocess

if __name__ == '__main__':
    download_process()
    Preprocess.proc_prepare()
