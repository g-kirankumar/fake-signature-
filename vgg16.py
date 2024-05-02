import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Function to preprocess an image for the VGG16 model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to calculate the similarity score between a reference image and a list of target images
def calculate_similarity_for_datasets(reference_image_paths, target_image_paths):
    reference_features = []

    for reference_image_path in reference_image_paths:
        reference_img = preprocess_image(reference_image_path)
        reference_features.append(model.predict(reference_img).flatten())

    similarities = []
    true_labels = []

    for target_image_path in target_image_paths:
        target_img = preprocess_image(target_image_path)
        target_features = model.predict(target_img).flatten()

        target_similarity_scores = []

        for reference_feature in reference_features:
            # Compute the cosine similarity between the feature vectors
            similarity = np.dot(reference_feature, target_features) / (np.linalg.norm(reference_feature) * np.linalg.norm(target_features))
            target_similarity_scores.append(similarity)

        max_similarity_score = max(target_similarity_scores)
        similarities.append(max_similarity_score)

        # Determine the true label based on the directory (1real or 1fake)
        true_label = 1 if '1real' in target_image_path else 0
        true_labels.append(true_label)

    return similarities, true_labels

# Directories containing the training and testing images
training_image_directory = "/content/drive/MyDrive/signatures/Train/real/500real"
testing_image_directory = "/content/drive/MyDrive/signatures/test/fake/500fake"

# List of training and testing image file names
training_image_files = os.listdir(training_image_directory)
testing_image_files = os.listdir(testing_image_directory)

# Reference image paths (training images)
reference_image_paths = [os.path.join(training_image_directory, file) for file in training_image_files]

# Calculate similarities for testing images against training images
similarities, true_labels = calculate_similarity_for_datasets(reference_image_paths, [os.path.join(testing_image_directory, file) for file in testing_image_files])

# You can set a threshold to determine if the images match or not
threshold = 0.80  # Adjust this threshold as needed

# Create binary predictions based on the threshold
predictions = [1 if score >= threshold else 0 for score in similarities]

# Calculate confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1fake', '1real'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

report = classification_report(true_labels, predictions)
print("Classification Report:")
print(report)
for target_image_path, similarity_score in zip([os.path.join(testing_image_directory, file) for file in testing_image_files], similarities):
    if similarity_score >= threshold:
        print(f"Image {target_image_path} is similar with a similarity score of: {similarity_score}\n")
    else:
        print(f"Image {target_image_path} is dissimilar with a similarity score of: {similarity_score}\n")