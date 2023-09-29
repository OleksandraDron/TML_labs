import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Зчитування даних з файлу
data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')
X = data[:, :-1]  # Параметри
y = data[:, -1]  # Результат

# Розділення даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Функція втрат: Logistic loss
def logistic_loss(y_true, y_pred):
    return tf.math.log1p(tf.exp(-y_true * y_pred))


# Функція втрат: Binary Crossentropy
def binary_crossentropy(y_true, y_pred):
    return -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))


# Функція втрат: Adaboost loss
def adaboost_loss(y_true, y_pred):
    return tf.exp(-y_true * y_pred)


# Функція для тренування та оцінки моделі з відповідною функцією втрат
def train_and_evaluate(loss_function, X_train, y_train, X_test, y_test, epochs=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)
    ])

    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    loss_train = history.history['loss']
    loss_test = history.history['val_loss']

    return model, loss_train, loss_test


# Тренування та оцінка моделей з різними функціями втрат
models = []
losses_train = []
losses_test = []
loss_functions = [logistic_loss, binary_crossentropy, adaboost_loss]

for loss_function in loss_functions:
    model, loss_train, loss_test = train_and_evaluate(loss_function, X_train, y_train, X_test, y_test)
    models.append(model)
    losses_train.append(loss_train)
    losses_test.append(loss_test)

# Візуалізація кривих навчання
plt.figure(figsize=(12, 6))
for i, loss_function in enumerate(loss_functions):
    plt.plot(losses_train[i], label=f"Train Loss ({loss_function})")
    plt.plot(losses_test[i], label=f"Test Loss ({loss_function})")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

# Порівняння якості класифікації за метрикою accuracy
accuracies = []

for model in models:
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

for i, loss_function in enumerate(loss_functions):
    print(f"Accuracy ({loss_function}): {accuracies[i]}")
