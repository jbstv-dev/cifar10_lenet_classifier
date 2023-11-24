from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def data_augmenter() -> ImageDataGenerator:
    """
    Data aumentation class
    Return:
        ImageDataGenerator class.
    """
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.125,
        height_shift_range=0.125,
        fill_mode="constant",
        cval=0.0,
    )

    return datagen


def load_data():
    """
    Load CIFAR10  dataset and performs normalization on images and one-hot encoding on labels
    Returns:
        x_train: images from training set
        y_train: lables in one-oht encoding for training set
        x_test: images from testing set
        y_test: lables in one-hot encoding for testing set
        classes: Dictionary containing each class label and its correspongin numerical label
    """
    print("[Info] Loading CIFAR10 dataset ...")
    # load CIFAR10 dataset.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Input image normalization.
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Encode categorical features as a one-hot numeric array.
    binarizer = LabelBinarizer()
    y_train = binarizer.fit_transform(y_train)
    y_test = binarizer.fit_transform(y_test)
    # Split train set into 80% training set and 20% validation set
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8)
    # Define classes names
    classes = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    print("[Info] Dataset loaded")
    print("number of training examples = " + str(x_train.shape[0]))
    print("number of test examples = " + str(x_test.shape[0]))
    print("X_train shape: " + str(x_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(x_test.shape))
    print("Y_test shape: " + str(y_test.shape))
    print("Classes: ", classes)

    return (x_train, y_train), (x_test, y_test), classes
