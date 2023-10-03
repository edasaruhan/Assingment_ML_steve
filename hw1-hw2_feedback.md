Hw1- The structure is correct. You can make your code more readible.
# normalization and gradient descent
X_scaled_max = normalize_max(X_train)
w_max, b_max = gradient_descent(X_scaled_max, y_train, alpha=0.01, num_iterations=1000)
y_pred_max = np.dot(normalize_max(X_test), w_max) + b_max
mse_max = calculate_mse(y_test, y_pred_max)
print("Mean Squared Error (Max):", mse_max)
You can use function. This is my suggestion.
Thank you for your efforts :)
