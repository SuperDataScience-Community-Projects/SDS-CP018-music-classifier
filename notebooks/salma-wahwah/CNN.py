from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # For saving scaler and label encoder

# Load the features and labels from the CSV file
features_df = pd.read_csv("notebooks/salma-wahwah/train/features.csv")

# Separate features (X) and labels (y)
X = features_df.drop(columns=['label']).values  # Drop 'label' column and take all features
y = features_df['label'].values  # 'label' column contains the target classes (genres)

# Encode the labels as integers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')
print("LabelEncoder saved successfully.")

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully.")

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Define the CNN model with reduced complexity and L2 regularization
model = Sequential()

# Reshape for Conv1D layer (since the input is 2D, we add an extra dimension for Conv1D)
model.add(Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)))

# Add a convolutional layer (1D) with L2 regularization and max-pooling
model.add(Conv1D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())  # Batch normalization to stabilize learning
model.add(MaxPooling1D(2))

# Add another Conv1D layer with fewer filters
model.add(Conv1D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling1D(2))

# Flatten and output layers
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))  # Dropout layer to prevent overfitting
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))  # Output layer with softmax for multi-class classification

# Compile the model with a higher learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summarize the model
model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the trained model
model.save('music_genre_cnn_model_regularized.h5')
print("Model saved successfully.")

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history for accuracy and loss
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
