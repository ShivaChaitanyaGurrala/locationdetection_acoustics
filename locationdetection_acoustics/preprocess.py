"""
    This should be used to get the audio recording and convert into spectrogram image and return
"""
import pathlib
import matplotlib.pyplot as plt
from csv import reader
from tqdm import tqdm
import librosa
import librosa.display
import numpy as np
import toml
from PIL import Image
# import tensorflow as tf
from scipy import signal
from skimage.transform import resize


def scale_min_max(img_arr, min_value, max_value):
    """

    :param img_arr:
    :param min_value:
    :param max_value:
    :return:
    """
    img_std = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
    image_scaled = img_std * (max_value - min_value) + min_value
    return image_scaled


def generate_spectrogram(f_path, label, folder_path):
    """

    :param f_path:
    :param label:
    :param folder_path:
    :return:
    """
    path_ = pathlib.Path.cwd() / "input_dir" / folder_path / label
    # removed locationdetection_acoustics/ from the beginning
    config = toml.load(
        pathlib.Path(r"../locationdetection_acoustics/locationdetection_acoustics/config.toml"))
    # if this is set we will generate spectrogram for audio files
    # print(type(config["preprocess"]["is_spectrogram"]))
    img = None
    if config["preprocess"]["is_spectrogram"] == 1:

        if not path_.is_dir():
            try:
                path_.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                print("Folder is already there")
            else:
                print("Folder was created")

        mels = config["preprocess"]["n_mels"]
        f_min = config["preprocess"]["fmin"]
        f_max = config["preprocess"]["fmax"]
        nfft = config["preprocess"]['n_fft']
        sr = config["preprocess"]["sr"]
        # for log melspectogram features
        if config["preprocess"]["manual"] == 0:
            # print("Generating Images using melspectogram ")
            waveform, sample_rate = librosa.load(f_path, offset=1.0, duration=10.0, sr=sr)
            # hop lenth is for each time to sample next window
            # n_fft is sample input window size
            # n_mels frequencies separated equally into these bins on human hearing capabilities
            img = librosa.feature.melspectrogram(y=waveform, sr=sr, hop_length=1024, n_mels=mels)
            # adding small number to avoid log(0)
            img = np.log(img + 1e-9)

            # min-max scale to fit inside 8-bit range
            img = scale_min_max(img, 0.0, 255).astype(np.uint8)
            img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
            # img = 255 - img  # invert. make black==more energy
            # img = Image.fromarray(img)

            # for mel with mfcc added
        elif config["preprocess"]["manual"] == 1:
            if config["preprocess"]['window'] == 'hamming_asymmetric':
                window = signal.windows.hamming(nfft, sym=False)
            else:
                window = "hann"
            waveform, sample_rate = librosa.load(f_path, offset=1.0, duration=10.0, sr=sr)
            # win length is for window length in seconds
            power_spectrogram = np.abs(librosa.stft(
                waveform + 1e-9,
                n_fft=nfft,
                win_length=int(config['preprocess']['win_length_seconds'] * config['preprocess']['sr']),
                hop_length=int(config['preprocess']['hop_length_seconds'] * config['preprocess']['sr']),
                center=True,
                window=window
            )) ** 2
            mel_basis = librosa.filters.mel(
                sr=44100,
                n_fft=nfft,
                n_mels=mels,
                fmin=f_min,
                fmax=f_max)
            mel_spectrum = np.dot(mel_basis, power_spectrogram)
            mfcc_img = librosa.feature.mfcc(S=librosa.amplitude_to_db(mel_spectrum),
                                            n_mfcc=config['preprocess']['n_mfcc'])

            # Delta coefficients
            mfcc_delta = librosa.feature.delta(mfcc_img)
            # Delta 2 coefficients
            mfcc_delta2 = librosa.feature.delta(mfcc_img, order=2)

            # Add Delta Coefficients to feature matrix

            # mfcc_img = np.vstack((mfcc_img, mfcc_delta, mfcc_delta2))
            mfcc_img = mfcc_img + mfcc_delta + mfcc_delta2
            mfcc_img = mfcc_img[1:, :]
            mfcc_img = np.flip(mfcc_img)
            img = mfcc_img
            # img = np.log(img + 1e-9)

            # min-max scale to fit inside 8-bit range
            # img = img - np.mean(img) / np.std(img)
            # img = scale_min_max(img, np.mean(img) - np.std(img), np.mean(img) + np.std(img))
            # img = np.flip(img, axis=0)
            # img = resize(img, (128, 256))
            # height, width = img.shape
            # print(img.shape)
            # img = img.reshape(1, height, width)
            # print(img.shape)
            # for chroma features only
        elif config["preprocess"]["manual"] == 4:
            waveform, sample_rate = librosa.load(f_path, offset=1.0, duration=10.0, sr=sr)
            mfccs = librosa.feature.mfcc(waveform, sr=sample_rate)
            img = mfccs[2:, :]
            img = np.flip(img)
        '''elif config["preprocess"]["manual"] == 2:
            audio = tfio.audio.AudioIOTensor(f_path)
            audio_tensor = audio.to_tensor()
            import pdb
            pdb.set_trace()
            # audio_tensor = tf.squeeze(audio, axis=[-1])
            tensor = tf.cast(audio_tensor, tf.float32) / 32768.0
            # position = tfio.experimental.audio.trim(tensor, axis=0, epsilon=0.1)
            # audio = tfio.experimental.audio.fade(
            #    processed, fade_in=1000, fade_out=2000, mode="logarithmic")
            spectrogram = tfio.experimental.audio.spectrogram(
                audio_tensor, nfft=512, window=512, stride=256)
            mel_spectrogram = tfio.experimental.audio.melscale(
                spectrogram, sr, mels, f_min, f_max, name=None
            )
            # Convert to db scale mel-spectrogram
            img = tfio.experimental.audio.dbscale(
                mel_spectrogram, top_db=80)'''


        # img = np.expand_dims(img, 0)
        # img = Image.fromarray(img)
        # img = np.array(Image)
        # if img.mode != 'RGB':
        #    img = img.convert('RGB')

        f_path = f_path.split("/")
        f_path = f_path[len(f_path) - 1].split(".")[0] + ".png"
        s_file_name = path_ / f_path
        # scipy.misc.imsave(s_file_name, img)
        # img.save(s_file_name)
        plt.imsave(s_file_name, img)
        # plt.close()
        # img.save(s_file_name)
    else:
        print(" Spectrogram's are generated on the fly using inbuilt mechanism ")


class Preprocess:
    def __init__(self):
        self.train_file = "fold1_train"
        self.test_file = "fold1_evaluate"
        self.path_ = pathlib.Path.cwd()
        # removed / "locationdetection_acoustics"  from the below path
        self.read_path = pathlib.Path(
            self.path_.parent / "locationdetection_acoustics" / "input_dir"
            / "TAU-urban-acoustic-scenes-2019-development_meta"
            / "TAU-urban-acoustic-scenes-2019-development" / "evaluation_setup")

    def proc_prepare(self):
        for file in self.read_path.iterdir():
            folder = None
            if self.train_file in file.as_posix():
                folder = "train"
            elif self.test_file in file.as_posix():
                folder = "test"
            if folder is not None:
                if file.is_file():
                    with open(file, "r") as read_obj:
                        csv_reader = reader(read_obj, delimiter="\t")
                        header = next(csv_reader)
                        # Check file as empty
                        if header is not None:
                            # Iterate over each row after the header in the csv
                            print(f"Generating Spectrogram's of {folder} Dataset")
                            for row in tqdm(csv_reader):
                                #
                                # f_name, f_label = row.split("\t")
                                search_file = "**/" + row[0]
                                # use pathlib glob() to fetch the audio file
                                audio_file_path = self.path_.parent.glob(search_file)
                                for x in list(audio_file_path):
                                    generate_spectrogram(x.as_posix(), row[1], folder)


if __name__ == "__main__":
    p = Preprocess()
    p.proc_prepare()
