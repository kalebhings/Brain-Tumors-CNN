#%%
# Import libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


#%%
# Import libraries
import os
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Function to prepend subdirectories to the image filenames in the dataframe
def add_subdirectory_path(row):
    subdirectory = "Brain_Tumor" if row['class'] == 'tumor' else "Healthy"
    return os.path.join(subdirectory, row['image'])

# Directories
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_directory, 'Brain Tumor Data Set')

# Load metadata
metadata_path = os.path.join(current_directory, 'metadata.csv')
metadata = pd.read_csv(metadata_path)

# Filter for JPEG images
metadata = metadata[metadata['format'] == 'JPEG']

# Add subdirectory paths to the image filenames
metadata['image'] = metadata.apply(add_subdirectory_path, axis=1)

# Simplify class labels to strings for binary classification
metadata['class'] = metadata['class'].replace({'tumor': 'tumor', 'healthy': 'healthy'})  # Keep the labels as strings

# Split the metadata into train, test, and validation sets
train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=42)
train_metadata, valid_metadata = train_test_split(train_metadata, test_size=0.2, random_state=42)

# Image Data Generators
# Updated Image Data Generators with Data Augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # degrees
    width_shift_range=0.1,  # fraction of total width
    height_shift_range=0.1, # fraction of total height
    shear_range=0.2,        # shear intensity (shear angle in degrees)
    zoom_range=0.2,         # amount of zoom
    horizontal_flip=True,   # flipping the images horizontally
    fill_mode='nearest'     # strategy to fill newly created pixels
)

# The test_datagen and valid_datagen remain unchanged because we shouldn't augment our validation and test sets
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Now, recreate the data generators with the updated datagen
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_metadata,
    directory=dataset_dir,
    x_col='image',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_metadata,
    directory=dataset_dir,
    x_col='image',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_metadata,
    directory=dataset_dir,
    x_col='image',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


# Define the CNN model
from tensorflow.keras import regularizers

from tensorflow.keras.layers import BatchNormalization

model = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3), kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),  # Add batch normalization
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),  # Add batch normalization
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(128, (3, 3), strides=(2, 2), kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),  # Add batch normalization
    layers.Activation('relu'),
    
    layers.Flatten(),
    layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    layers.Activation('swish'),
    
    layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    layers.Activation('swish'),
    
    layers.Dense(1, activation='sigmoid')
])


model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#%%
# Train the model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Add callbacks to the fit method
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_metadata) // 32,
    epochs=50,  # Increase epochs
    validation_data=valid_generator,
    validation_steps=len(valid_metadata) // 32,
    callbacks=[early_stopping, model_checkpoint]
)


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.2f}")


#%%
# First, we need to get the true labels from the test set
# Since we have used 'flow_from_dataframe', the generator will have the labels in the same order as the dataframe
true_labels = test_metadata['class'].map({'tumor': 1, 'healthy': 0}).values

# Generate predictions
# Make sure the generator is not shuffling the data
test_generator.reset()  # Ensure we are starting from the beginning
predictions = model.predict(test_generator)
predicted_classes = np.round(predictions).astype(int).flatten()  # Convert probabilities to binary class labels

# Calculate the classification report
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(true_labels, predicted_classes))

# Calculate overall accuracy
accuracy = accuracy_score(true_labels, predicted_classes)
print(f"Accuracy: {accuracy:.2f}")

# Identify misclassified examples
misclassified_indices = np.where(predicted_classes != true_labels)[0]
print(f"Total misclassified images: {len(misclassified_indices)}")

# Optional: Visualize some of the misclassified examples
# Note: This will work only if the test_generator has not been shuffled.
if len(misclassified_indices) > 0:
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(misclassified_indices[:9]):  # show up to 9 misclassified examples
        # Get the image from the test_generator
        img, _ = test_generator[idx]
        img = img[0]  # Get the image from the batch
        actual_label = 'tumor' if true_labels[idx] == 1 else 'healthy'
        predicted_label = 'tumor' if predicted_classes[idx] == 1 else 'healthy'
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(f'Actual: {actual_label}\nPredicted: {predicted_label}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No misclassified images to display.")

#%%
# Check for NaN values in the class column
print("Number of NaN values in 'class' column:", metadata['class'].isnull().sum())
#%%

# Drop rows with NaN values in the 'class' column
metadata.dropna(subset=['class'], inplace=True)
#%%
# Confirm that all classes are mapped correctly
metadata['label'] = metadata['class'].map({'tumor': 1, 'healthy': 0})
print("Unique classes after mapping:", metadata['label'].unique())

#%%
# Generate predictions
test_generator.reset()  # reset the generator to be sure
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.round(predictions).astype(int).flatten()

# Confirm matching length
assert len(predicted_classes) == len(true_labels), "Mismatch between length of predictions and true labels"
# Generate the classification report
print(classification_report(true_labels, predicted_classes))


#%%
from sklearn.metrics import classification_report, accuracy_score

# Obtain true labels from metadata
true_labels = metadata['class'].map({'tumor': 1, 'normal': 0}).values

# Get predicted classes from your model
# Assuming you have predictions stored in the variable predicted_classes

# Calculate the classification report
print(classification_report(true_labels, predicted_classes))

# Calculate overall accuracy
accuracy = accuracy_score(true_labels, predicted_classes)
print("Overall accuracy:", accuracy)
