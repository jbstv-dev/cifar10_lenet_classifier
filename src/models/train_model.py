import argparse
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    LearningRateScheduler,
)
from src.models.visualize import (
    plot_metrics,
    display_confusion_matrix,
    plotmodel,
    visualize_layers,
    visualize_predictions,
)
from src.data.make_dataset import load_data, data_augmenter
from src.models.predict_model import report, make_prediction
from src.models.models import lenet5

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(
    "-e",
    "--epochs",
    default=100,
    type=int,
    help="Number of epochs to train data, default=100",
)
argument_parser.add_argument(
    "-m",
    "--model",
    default="lenet5",
    type=str,
    help="Model type: LeNet5 or AlexNet, default=AlexNet",
)
argument_parser.add_argument(
    "-b", "--batch", default=120, type=int, help="Number of images per batch"
)
argument_parser.add_argument(
    "-opt",
    "--optimizer",
    default="adam",
    type=str,
    help="Optimizer algorithm, dafault=adam",
)
arguments = vars(argument_parser.parse_args())


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-2
    lr = lr * (0.1 ** int(epoch / 15))
    print("Learning rate: ", lr)
    return lr


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.device(gpus[0])

    # load data
    (x_train, y_train), (x_test, y_test), classes = load_data()

    # Load model
    model = lenet5(x_train[1].shape, y_train.shape[1])

    # Define hyperparameters
    batch_size = 128
    epochs = 200
    iterations = 391
    model_type = "LeNet"

    # data augmentation
    datagen = data_augmenter()

    # change_lr = LearningRateScheduler(scheduler)
    log_csv = CSVLogger(
        os.path.join(os.getcwd(), "/results/logs_%s.csv" % model_type),
        separator=",",
        append=False,
    )
    cbks = [log_csv]
    # start train
    print("[Info] Training model ...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        steps_per_epoch=iterations,
        epochs=epochs,
        callbacks=cbks,
        validation_data=(x_test, y_test),
        verbose=1,
    )
    # save model
    save_dir = os.getcwd() + "/models"
    filepath = os.path.join(save_dir, "lenet.h5")
    model.save(filepath)

    print("[Info] Saving model history")

    # y_val_pred = model.predict(x_val, batch_size=arguments["batch"])
    # y_train_pred = model.predict(x_train, batch_size=arguments["batch"])
    hist = "logs_%s.csv" % model_type
    model_name = "lenet.h5"
    plot_metrics(hist, model_type)
    display_confusion_matrix(x_test, y_test, classes, model_name)
    report(x_test, y_test, classes, model_name)
