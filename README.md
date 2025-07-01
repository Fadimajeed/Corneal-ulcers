# AI-Powered Eye Disease Classifier

## Project Overview

This project presents an advanced desktop application designed for the automated classification of eye diseases from slit-lamp images. Leveraging a sophisticated pipeline of image preprocessing, deep learning for feature extraction, and ensemble machine learning for classification, this tool aims to assist ophthalmologists and healthcare professionals in the rapid and accurate diagnosis of various corneal conditions. The application features a user-friendly graphical interface, real-time image analysis, and provides detailed diagnostic results along with estimated damage assessments and actionable recommendations.
## Dataset url 
https://www.kaggle.com/datasets/bongsang/eye-disease-deep-learning-dataset
## Features

*   **Automated Disease Classification:** Accurately classifies eye diseases from slit-lamp images using a trained machine learning model.
*   **Advanced Image Preprocessing:** Implements CLAHE (Contrast Limited Adaptive Histogram Equalization), center cropping, and resizing to optimize image quality for analysis.
*   **Deep Learning Feature Extraction:** Utilizes a pre-trained TensorFlow/Keras model to extract robust features from processed images.
*   **Ensemble Machine Learning:** Employs an XGBoost classifier for highly accurate and reliable disease prediction.
*   **Cornea Validation:** Includes intelligent checks to determine if an uploaded image is a valid fluorescein-stained corneal image, preventing erroneous classifications.
*   **Damage Estimation & Recommendations:** Provides a calculated damage percentage and tailored recommendations based on the classification results.
*   **Intuitive Graphical User Interface (GUI):** Developed with CustomTkinter, offering a modern and responsive interface for seamless user interaction.
*   **Real-time Processing & Feedback:** Displays image previews, processing progress, and immediate analysis results.
*   **Analysis History:** Maintains a log of past analyses for easy review.

## Technologies Used

*   **Python:** The core programming language for the application.
*   **CustomTkinter:** For building the modern and cross-platform desktop GUI.
*   **TensorFlow/Keras:** For deep learning model loading and feature extraction from images.
*   **scikit-learn:** For machine learning utilities, including `joblib` for model persistence.
*   **XGBoost:** The primary ensemble learning model used for classification.
*   **OpenCV (cv2):** For advanced image processing operations, including CLAHE, image manipulation, and Hough Circle detection.
*   **PIL (Pillow):** For image handling and display within the GUI.
*   **NumPy:** For numerical operations and array manipulation.
*   **Threading:** To ensure a responsive UI during image processing.

## Getting Started

To set up and run the Eye Disease Classifier locally, follow these steps:

### Prerequisites

*   Python 3.9
*   Required Python packages (listed below)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Fadimajeed/Corneal-ulcers.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd eye_disease_classifier
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r customtkinter tensorflow scikit-learn xgboost opencv-python Pillow numpy
    ```

4.  **Place Model Files:**
    Ensure `eye_disease_classifier.pkl` and `eye_disease_classifier_tfmodel.keras` are in the root directory of the project.

### Usage

To run the application, execute the `interface.py` file:

```bash
python interface.py
```

Once the application launches, click the "Upload Slit-Lamp Image" button to select an image for analysis. The application will process the image, classify the eye disease, and display the results along with recommendations.

## Project Structure

```
.
├── interface.py          
├── backend.py             
├── eye_disease_classifier.pkl  
├── eye_disease_classifier_tfmodel.keras 
```

## Model Details

The classification pipeline consists of two main stages:

1.  **Feature Extraction:** A trained TensorFlow/Keras model (`eye_disease_classifier_tfmodel.keras`) is used to extract high-level features from the preprocessed slit-lamp images. This model acts as a powerful feature extractor, converting raw image data into a numerical representation suitable for traditional machine learning algorithms.
2.  **Classification:** The extracted features are then fed into an XGBoost classifier (`eye_disease_classifier.pkl`). XGBoost, an ensemble learning method, is highly effective for classification tasks and provides robust predictions based on the features. The model also includes logic for confidence thresholds and category-specific probability checks.

### Image Preprocessing

Images undergo a series of preprocessing steps to enhance relevant features and standardize input for the models:

*   **CLAHE:** Applied to improve local contrast, especially in images with varying lighting conditions.
*   **Center Cropping:** Ensures that the central, most relevant part of the eye is focused on.
*   **Resizing:** Images are resized to a consistent `768x768` resolution.

### Corneal Validation

Before classification, the application performs checks to ensure the uploaded image is indeed a fluorescein-stained cornea:

*   **Fluorescein Color Check:** Analyzes the image for a significant presence of green/blue hues typical of fluorescein under cobalt-blue light.
*   **Corneal Circle Detection:** Uses Hough Circle Transform to identify a circular region, confirming the presence of a cornea.
## Images
![image](https://github.com/Fadimajeed/Corneal-ulcers/blob/master/py-images/Results.jpg?raw=true)
![image](https://github.com/Fadimajeed/Corneal-ulcers/blob/master/py-images/Screenshot%202025-05-12%20140042.png?raw=true)
![image](https://github.com/Fadimajeed/Corneal-ulcers/blob/master/py-images/Screenshot%202025-05-12%20140208.png?raw=true)
![image](https://github.com/Fadimajeed/Corneal-ulcers/blob/master/py-images/Screenshot%202025-05-12%20140232.png?raw=true)
![image](https://github.com/Fadimajeed/Corneal-ulcers/blob/master/py-images/Screenshot%202025-05-12%20140401.png?raw=true)
![image](https://github.com/Fadimajeed/Corneal-ulcers/blob/master/py-images/Screenshot%202025-05-12%20142604.png?raw=true)

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


