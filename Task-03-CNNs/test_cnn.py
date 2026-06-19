import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
model = tf.keras.models.load_model("/Users/akulbakshi/Desktop/ml notes/all open cv/week 3/cnn_model.h5")
y_test = to_categorical(y_test,10)
test_loss, test_accuracy = model.evaluate(x_test, y_test,verbose = 0)

y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

with open("cnn_evaluation_output.txt", "w") as f:
    f.write(f"test accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"test loss: {test_loss:.4f}\n\n")
    f.write("classification report:\n")
    f.write(classification_report(np.argmax(y_test, axis=1), y_pred))
    f.write("\nconfusion matrix:\n")
    f.write(str(confusion_matrix(np.argmax(y_test, axis=1), y_pred)))
