import tensorflow as tf
import toml
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
from models import RCNN
from tensorflow.keras.optimizers import Adam
import numpy as np


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


# print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Load Configuration fields.
config = toml.load(
    pathlib.Path(r"/home/shiva/Desktop/locationdetection_acoustics/locationdetection_acoustics/config.toml"))
config = config["preprocess"]
# 1 - Load the model and its pretrained weights if exists
# classifier = cnn()
# classifier.load('weights/cnn_DF')
model = RCNN(kernel_size=config["kernel_size"], filters=config["filters"])
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(
                  learning_rate=config['lr'], beta_1=config['beta_1'],
                  beta_2=config['beta_2'], epsilon=1e-7), metrics=['accuracy'])

# let us use ImageDataGenerator to pick data to be trained
curr = pathlib.Path.cwd()
train_path = pathlib.Path(curr.parent / "input_dir" / "train")
test_path = pathlib.Path(curr.parent / "input_dir" / "test")

# define train data generator
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    width_shift_range=0.6,
    preprocessing_function=get_random_eraser(v_l=0, v_h=255)
)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(256, 388),  # All images will be resized to 256x388
    batch_size=config['batch_size'],
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True
)

# define test data generator
test_datagen = ImageDataGenerator(
    featurewise_center=True
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(256, 388),  # All images will be resized to 150x150
    batch_size=config['batch_size'],
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True
)

# start training
history = model.fit(
    train_generator,
    steps_per_epoch=575,
    epochs=50,
    verbose=1,
    validation_data=test_generator,
    validation_steps=262
)

model.save('newweights/Conv5.h5')
