# part 1 - data preparation
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# part 2 - build models 
# model 1 
model_1 = build_model(1, 64)
model_1.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2) 

# model 2 
model_2 = build_model(1, 128)
model_2.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# model 3 
model_3 = build_model(1, 32)
model_3.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# part 3 - model evaluation 
loss_1 = model_1.evaluate(x_test, y_test)
loss_2 = model_2.evaluate(x_test, y_test)
loss_3 = model_3.evaluate(x_test, y_test)


plt.scatter(x_test, y_test, color='blue', label='True Function')
plt.scatter(x_test, model_1.predict(x_test), color='red', label='Model 1 Predictions')
plt.scatter(x_test, model_2.predict(x_test), color='green', label='Model 2 Predictions')
plt.scatter(x_test, model_3.predict(x_test), color='yellow', label='Model 3 Predictions')

plt.legend()
plt.title('Model Predictions')

# part 4 - get model output and feedforward by yourself
best_model = model_1 if loss_1 < loss_2 and loss_1 < loss_3 else model_2 if loss_2 < loss_3 else model_3

chosen_indices = np.random.choice(len(x_train), 5)
chosen_data = x_train[chosen_indices]

model_predictions = best_model.predict(chosen_data)

for i in range(len(chosen_data)):
    print("Manual Calculation Result:", manual_results[i][0])
    print("Model Prediction Result:", model_predictions[i][0])
