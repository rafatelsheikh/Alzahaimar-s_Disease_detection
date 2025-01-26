import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import time


def otsu_threshold_opencv(image):
  """
  Finds the optimal threshold for image using OpenCV's Otsu thresholding function.

  Args:
      image: A 2D grayscale image represented as a numpy array.

  Returns:
      The binary image with the applied threshold.
  """
  # Convert image to grayscale if needed (assuming RGB image)
  try:
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  except AttributeError:
    print("error")

  # Apply Otsu thresholding
  _, thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  return thresh_img


def read_images(folder_path):
  features = []
  for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter for image formats
      image_path = os.path.join(folder_path, filename)
      original_image = cv2.imread(image_path)
      thresholded_image = otsu_threshold_opencv(original_image)
      features.append(extract_features(original_image, thresholded_image))

  return features


def calculate_perimeter(image):
  """
  This function calculates the perimeter of the largest connected component (assumed to be brain) in a grayscale image.

  Args:
      image: A grayscale image representing the MRI scan.

  Returns:
      The perimeter of the largest connected component, or 0 if no component is found.
  """

  # Find contours (boundary of connected components)
  contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Find the largest contour (assuming it corresponds to the brain)
  largest_contour = None
  largest_area = 0
  for contour in contours:
    area = cv2.contourArea(contour)
    if area > largest_area:
      largest_contour = contour
      largest_area = area

  # Calculate perimeter if a large enough contour is found
  perimeter = 0
  if largest_contour is not None:
    # Approximate the contour with straight lines for simpler perimeter calculation
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    perimeter = cv2.arcLength(approx, True)

  return perimeter


def extract_features(original_image, thresholded_image):
  # Total Area (number of white pixels)
  total_area = np.sum(thresholded_image == 255)

  # Mean Intensity (assuming original image had intensity information)
  mean_intensity = np.mean(original_image[thresholded_image == 255])

  # Standard deviation (assuming original image had intensity information)
  std_dev = np.std(original_image[thresholded_image == 255])

  # Calculate the perimeter
  perimeter = calculate_perimeter(thresholded_image)

  features = [total_area, mean_intensity, std_dev, perimeter]

  return features


def kfold_evaluate(k, features, labels):
    # Initialize KFold object
    kf = KFold(n_splits=k, shuffle=True)  # Shuffle data for better randomization

    # Track overall accuracy
    total_accuracy = 0

    # Iterate through folds
    for train_index, test_index in kf.split(features):
        # Split data into training and testing sets
        train_features = [features[i] for i in train_index]
        train_labels = [labels[i] for i in train_index]
        test_features = [features[i] for i in test_index]
        test_labels = [labels[i] for i in test_index]

        # Train the model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(train_features, train_labels)

        # Make predictions on test set
        predictions = model.predict(test_features)

        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)
        total_accuracy += accuracy

    # Calculate average accuracy across folds
    average_accuracy = total_accuracy / k
    print(f"Average Accuracy (k={k} folds): {average_accuracy:.4f}")
    return model


# Data preparation
features_AD_train = read_images(r"E:\Data set\archive\Alzheimer_s Dataset\train\AD")
length_AD_train = len(features_AD_train)
labels_train = ["AD"] * length_AD_train

features_healthy_train = read_images(r"E:\Data set\archive\Alzheimer_s Dataset\train\NonDemented")
length_healthy_train = len(features_healthy_train)
labels_train += ["Healthy"] * length_healthy_train

features_train = []
features_train.extend(features_AD_train)
features_train.extend(features_healthy_train)


features_AD_test = read_images(r"E:\Data set\archive\Alzheimer_s Dataset\test\AD")
length_AD_test = len(features_AD_test)
labels_test = ["AD"] * length_AD_test

features_healthy_test = read_images(r"E:\Data set\archive\Alzheimer_s Dataset\test\NonDemented")
length_healthy_test = len(features_healthy_test)
labels_test += ["Healthy"] * length_healthy_test

features_test = []
features_test.extend(features_AD_test)
features_test.extend(features_healthy_test)

# Model training
start_time = time.time()

model = kfold_evaluate(5, features_train, labels_train)

train_time = time.time() - start_time
joblib.dump(model, 'trained_model7.pkl')

start_time = time.time()

# Model testing
predictions = model.predict(features_test)
accuracy = accuracy_score(labels_test, predictions)

# Generate confusion matrix
confusion_matrix = confusion_matrix(labels_test, predictions)
precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])

# Calculate recall for the positive class (assuming class labels are "AD" and "Healthy")
recall = recall_score(labels_test, predictions, pos_label="AD")

# Calculate F1 score for the positive class (assuming class labels are "AD" and "Healthy")
f1 = f1_score(labels_test, predictions, pos_label="AD")

test_time = time.time() - start_time

print("Model Accuracy:", accuracy)
print("Precision for AD class:", precision)
print("Recall for AD class:", recall)
print("F1 score for AD class:", f1)
print("Training time: ", train_time)
print("Testing time: ", test_time)





"""
# Example usage
image = cv2.imread("E:\Healthy\magnetic-resonance-image-mri-normal-260nw-1657054951-ezgif.com-webp-to-jpg-converter.jpg")  # Load your image
thresholded_image = otsu_threshold_opencv(image)

# Now you can display or save the thresholded_image 
cv2.imshow("Otsu Thresholding", thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
