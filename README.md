import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import cv2

# Sample dataset (Replace with your labeled dataset)
data = pd.DataFrame({
    'Image_Path': ['path_to_image1.jpg', 'path_to_image2.jpg', ...],
    'Personality_Trait': ['Introvert', 'Extrovert', ...]
})

# Feature extraction (Using fictional features)
def extract_features(image_path):
    # Load and preprocess image (resize, normalize, etc.)
    image = cv2.imread(image_path)
    # Extract and compute features (e.g., fictional facial feature extraction)
    features = np.random.rand(10)  # Replace with actual feature extraction logic
    return features

data['Features'] = data['Image_Path'].apply(extract_features)

# Split the dataset into training and test sets
X = np.vstack(data['Features'])
y = data['Personality_Trait']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple classifier (Random Forest as an example)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict personality traits
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Make predictions for new data
new_data = pd.DataFrame({
    'Image_Path': ['path_to_new_image.jpg'],
    'Features': [extract_features('path_to_new_image.jpg')]
})
predicted_personality_trait = classifier.predict(np.vstack(new_data['Features']))[0]

print(f"Predicted Personality Trait: {predicted_personality_trait}")
