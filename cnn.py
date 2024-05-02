import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
# Define the image dimensions and batch size
image_width = 64
image_height = 64
batch_size = 32

# Specify the paths to your training and testing data directories
train_data_dir = "/content/drive/MyDrive/signatures/Train/real"
test_data_dir = "/content/drive/MyDrive/signatures/test/fake"

# Use data augmentation for training data to improve model generalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,       # Normalize pixel values to [0, 1]
    rotation_range=40,         # Randomly rotate images
    width_shift_range=0.2,    # Randomly shift images horizontally
    height_shift_range=0.2,   # Randomly shift images vertically
    shear_range=0.2,          # Shear transformations
    zoom_range=0.2,           # Randomly zoom in on images
    horizontal_flip=True,     # Randomly flip images horizontally
    fill_mode='nearest'       # Fill in newly created pixels
)

# Use simple rescaling for the testing data (no data augmentation)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load and prepare the training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary'  # Use binary labels for cats and dogs
)

# Load and prepare the testing data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Define the CNN model
model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(train_generator, steps_per_epoch=len(train_generator), validation_data=test_generator, validation_steps=len(test_generator), epochs=10)

predipredictions = model.predict(test_generator)

p_max = max(predipredictions)
p_min = min(predipredictions)

print(f"p max = {p_max}, p min = {p_min}")

t = ((p_max + p_min)) / 2

print(t)

for i in predipredictions:
    if i >= t:
        print(f"Match {i}")
    else:
        print(f"not matched {i}")

predicted_classes = (predipredictions > t).astype(int)

# True labels
true_classes = test_generator.classes

# Compute confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Display the confusion matrix
print("Confusion Matrix:")
print(cm)

# Calculate precision, recall, and F1-score
report = classification_report(true_classes, predicted_classes)
print("Classification Report:")
print(report)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])

cm_display.plot()
plt.show()