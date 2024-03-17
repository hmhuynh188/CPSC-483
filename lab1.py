# part 1 - data preparation
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Generate Training Data
x_values = np.linspace(-100, 100, num=100)
y_values = x_values * np.sin(x_values * 2300)

def build_model(input_shape, hidden_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values)

# part 2 - build models 
# model 1 
model_1 = build_model(1, 64)
model_1.fit(x_train, y_train)

# model 2 
model_2 = build_model(1, 128)
model_2.fit(x_train, y_train)

# model 3 
model_3 = build_model(1, 32)
model_3.fit(x_train, y_train)

# part 3 - model evaluation 
plt.scatter(x_test, y_test, color='blue', label='True Function')
plt.scatter(x_test, model_1.predict(x_test), color='red', label='Model 1 Predictions')
plt.scatter(x_test, model_2.predict(x_test), color='green', label='Model 2 Predictions')
plt.scatter(x_test, model_3.predict(x_test), color='yellow', label='Model 3 Predictions')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Predictions Comparison')

# part 4 - get model output and feedforward by yourself
best_model = model_1 if loss_1 < loss_2 and loss_1 < loss_3 else model_2 if loss_2 < loss_3 else model_3
chosen_indices = np.random.choice(len(x_train), 5)
chosen_data = x_train[chosen_indices]
model_predictions = best_model.predict(chosen_data)
chosen_data = x_train[chosen_indices]
for i in range(len(chosen_data)):
    print("Model Prediction Result:", model_predictions[i][0])
