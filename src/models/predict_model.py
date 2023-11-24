import os
from sklearn.metrics import classification_report
from keras.models import load_model
import numpy as np
import pandas as pd


def report(x_test, y_test, classes, model_name):
    path = os.getcwd() + "/results"
    model_path = os.getcwd() + "/models/"
    loaded_model = load_model(model_path + model_name)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(loaded_model.predict(x_test), axis=1)

    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )
    # print(report)

    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv(
        os.path.join(path, "classification_report_%s.csv" % model_name), index=True
    )


def make_prediction(x_test, y_test, classes, model_name):
    path = os.getcwd() + "/results"
    model_path = os.getcwd() + "/models"
    loaded_model = load_model(os.path.join(model_path, model_name))
    # save data in a dataframe
    predictions = loaded_model.predict(x_test)
    data = np.vstack(
        (
            np.argmax(y_test, axis=1),
            np.argmax(predictions, axis=1),
            np.max(predictions, axis=1),
        )
    ).T
    df = pd.DataFrame(data, columns=["Real", "Prediction", "Probability"])
    real = [classes[np.argmax(y_test, axis=1)[i]] for i in range(y_test.shape[0])]
    pred = [classes[np.argmax(predictions, axis=1)[i]] for i in range(y_test.shape[0])]
    df.insert(1, "Real Class", real)
    df.insert(3, "Predict Class", pred)
    df.to_csv(os.path.join(path, "predictions_%s.csv" % model_name), sep=",")
    print(df.head(20))
