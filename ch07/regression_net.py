import os
import urllib.request
from zipfile import ZipFile
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/"
    zip_file = "AirQualityUCI.zip"
    file_name = "AirQualityUCI.csv"
    data_root = os.path.join(os.path.dirname(__file__), "data")
    file_path = os.path.join(data_root, file_name)

    if not os.path.isfile(file_path):
        print("Download the data for regression...")
        url += zip_file
        zip_path = os.path.join(data_root, zip_file)
        urllib.request.urlretrieve(url, zip_path)
        with ZipFile(zip_path) as z:
            z.extract(file_name, data_root)
        os.remove(zip_path)

    return file_path


def load_dataset(file_path):
    dataset = pd.read_csv(file_path, sep=";", decimal=",")

    # Drop nameless columns
    unnamed = [c for c in dataset.columns if "Unnamed" in c]
    dataset.drop(unnamed, axis=1, inplace=True)

    # Drop unused columns
    dataset.drop(["Date", "Time"], axis=1, inplace=True)

    # Fill NaN by its column mean
    dataset.fillna(dataset.mean(), inplace=True)

    # Separate the data to label and features
    X = dataset.drop(["C6H6(GT)"], axis=1).values
    y = dataset["C6H6(GT)"].values.reshape(-1, 1)  # get benzene values
    return X, y


def make_model(input_size):
    inputs = Input(shape=(input_size,))
    hidden = Dense(8, activation="relu", kernel_initializer="glorot_uniform")
    output = Dense(1, kernel_initializer="glorot_uniform")

    pred = output(hidden(inputs))
    model = Model(inputs=[inputs], outputs=[pred])
    return model


def main():
    file_path = download_data()
    X, y = load_dataset(file_path)

    # Normalize the numerical values
    yScaler = StandardScaler()
    xScaler = StandardScaler()
    y = yScaler.fit_transform(y)
    X = xScaler.fit_transform(X)

    # Split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Make model
    input_size = X.shape[1]  # number of features
    model = make_model(input_size)

    # Train model
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    NUM_EPOCHS = 20
    BATCH_SIZE = 10
    model.compile(loss="mse", optimizer="adam")
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
        validation_split=0.2,
        callbacks=[TensorBoard(log_dir=log_dir)])

    # Make prediction
    y_pred = model.predict(X_test)

    # Show prediction
    y_pred = yScaler.inverse_transform(y_pred)
    y_test = yScaler.inverse_transform(y_test)
    result = pd.DataFrame({
        "prediction": pd.Series(y_pred.flatten()),
        "actual": pd.Series(y_test.flatten())
        })

    fig, ax = plt.subplots(nrows=2)
    ax0 = result.plot.line(ax=ax[0])
    ax0.set(xlabel="time", ylabel="C6H6 concentrations")
    diff = result["prediction"].subtract(result["actual"])
    ax1 = diff.plot.line(ax=ax[1], colormap="Accent")
    ax1.set(xlabel="time", ylabel="difference")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
