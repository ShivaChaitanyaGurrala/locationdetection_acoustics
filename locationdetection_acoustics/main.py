from downloader import download_process
from preprocess import Preprocess
from train import LocationClassifier
import toml
import pathlib

if __name__ == '__main__':
    config = toml.load(
        pathlib.Path(r"../locationdetection_acoustics/locationdetection_acoustics/config.toml"))
    if config["load"] == 0:
        download_process()
        prepare = Preprocess()
        prepare.proc_prepare()
    classifier = LocationClassifier()
    classifier.train()
