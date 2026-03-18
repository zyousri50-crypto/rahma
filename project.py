import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#: Set the seed for reproducible results
tf.random.set_seed(42)
# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Path to your Kaggle data (Adjust based on your specific Kaggle folder name)
base_path = r"C:\Users\Zakaria Yousri\Downloads\archive"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Automatically reserve 20% for testing
)

train_gen = train_datagen.flow_from_directory(
    base_path + '/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    base_path + '/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
def build_model(num_classes):
    # 1. Base: Pre-trained on ImageNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Freeze the base

    # 2. Custom Head: The "Quality Expert"
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x) # Prevents the model from "memorizing" images (overfitting)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model(train_gen.num_classes)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    verbose=1
)

# Save the model so you can use it later
model.save('fruit_quality_model.h5')

def check_produce(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    # Map index back to class name
    class_labels = {v: k for k, v in train_gen.class_indices.items()}
    result = class_labels[class_idx]
    confidence = prediction[0][class_idx] * 100

    print(f"I am {confidence:.2f}% sure this is: {result}")
    plt.imshow(img)
    plt.title(f"Result: {result}")
    plt.show()

# Example usage:
# check_produce('path_to_your_test_image.jpg')