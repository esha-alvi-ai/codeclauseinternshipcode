import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import random

# --------------------------
# 1. Load and Prepare MNIST
# --------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode the labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# --------------------------
# 2. Build a CNN Model
# --------------------------
model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------------
# 3. Train the Model
# --------------------------
model.fit(x_train, y_train_cat, epochs=5, batch_size=128, validation_split=0.1)

# --------------------------
# 4. Evaluate the Model
# --------------------------
loss, acc = model.evaluate(x_test, y_test_cat)
print(f"Test Accuracy: {acc * 100:.2f}%")

# --------------------------
# 5. Simulate Sequence Prediction
# --------------------------
def generate_digit_sequence(length=5):
    digits = []
    images = []
    for _ in range(length):
        idx = random.randint(0, len(x_test) - 1)
        digits.append(y_test[idx])
        images.append(x_test[idx])
    return digits, images

def predict_sequence(images):
    sequence = ""
    for img in images:
        prediction = model.predict(img.reshape(1, 28, 28, 1), verbose=0)
        digit = np.argmax(prediction)
        sequence += str(digit)
    return sequence

# --------------------------
# 6. Test Sequence Recognition
# --------------------------
true_digits, digit_images = generate_digit_sequence(5)
predicted_digits = predict_sequence(digit_images)

# Display
print("True digits:", ''.join(map(str, true_digits)))
print("Predicted  :", predicted_digits)

# Show images
plt.figure(figsize=(10, 2))
for i, img in enumerate(digit_images):
    plt.subplot(1, len(digit_images), i + 1)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Sequence of Handwritten Digits")
plt.show()
