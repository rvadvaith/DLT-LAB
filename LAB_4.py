# Import required libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize the data (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Define Feed Forward Neural Network model
model = Sequential([
    Flatten(input_shape=(28, 28)),          # Input layer (784 neurons)
    Dense(128, activation='relu'),          # Hidden layer
    Dense(10, activation='softmax')         # Output layer (10 classes)
])

# 5. Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train the model
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=5,
                    batch_size=128)

# 7. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# 8. Prediction Example
import matplotlib.pyplot as plt
import numpy as np

index = 10
plt.imshow(x_test[index], cmap='gray')
plt.title("Actual Label: " + str(np.argmax(y_test[index])))
plt.show()

prediction = model.predict(x_test[index].reshape(1,28,28))
print("Predicted Label:", np.argmax(prediction))
