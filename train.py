import json
import pathlib
import tensorflow as tf
from tensorflow.keras import layers, models

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define MobileNetV2 Transfer Learning architecture
def build_model():
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Resize images to 96x96 to better utilize ImageNet weights
    x = layers.Resizing(96, 96)(inputs)
    
    # Scale from [0, 1] to [-1, 1] as expected by MobileNetV2
    x = layers.Rescaling(scale=2.0, offset=-1.0)(x)
    
    # Load pretrained MobileNetV2 without the top classification layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base model to train only the top layers
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='cifar10_mobilenetv2')
    return model

model = build_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

# Create directories if not exist
model_dir = pathlib.Path('model')
model_dir.mkdir(parents=True, exist_ok=True)

# Save model
model_path = model_dir / 'cifar10_cnn.keras'
model.save(model_path)

# Save training history as JSON
history_path = model_dir / 'training_history.json'
with open(history_path, 'w') as f:
    json.dump(history.history, f)

print('Eğitim tamamlandı! Model kaydedildi:', model_path)
print('Eğitim geçmişi kaydedildi:', history_path)
