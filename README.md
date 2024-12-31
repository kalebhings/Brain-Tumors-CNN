# Brain Tumor Detection using MRI Images

This repository contains the implementation of a machine learning project aimed at predicting brain tumors from MRI images using the InceptionV3 model with transfer learning. The project includes data preprocessing, model training, and analysis of results.


## Project Overview
Brain tumor detection is a critical task in medical imaging. This project uses MRI images to predict the presence of brain tumors, leveraging the power of transfer learning with the InceptionV3 model to achieve high accuracy and robust performance. Here is a [link](https://github.com/kalebhings/Brain-Tumors-CNN/blob/main/brain_tumors_cnn.ipynb) to the notebook where the code can be viewed and the results of it running in the notebook.

## Data Preprocessing
- **Dataset:** The data used in this project was sourced from [Kaggle's Brain Tumor Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset).
- **Organization and Labeling:** MRI images were labeled and organized into categories (tumor and non-tumor).
- **Resizing and Normalization:** Images were resized to 299x299 pixels (compatible with InceptionV3) and normalized with pixel values between 0 and 1 for faster training convergence.
- **Data Augmentation:** To enhance robustness and generalization, data augmentation techniques were applied:
  - Rotation
  - Width and height shifting
  - Shear transformation
  - Zooming
  - Horizontal and vertical flipping

## Model Selection and Training
- **Transfer Learning with InceptionV3:** The InceptionV3 model was used due to its pre-trained capabilities on large datasets, enabling it to capture complex image features effectively. 
- **Comparison with Custom CNN:** A custom CNN model achieved 89% accuracy, while InceptionV3 outperformed it with 98% accuracy due to its robust architecture and feature extraction capabilities.
- **Dataset Size:** The dataset consisted of approximately 4,000 images, which was insufficient for a custom model to perform well but suited transfer learning due to the pre-trained nature of InceptionV3.

## Results and Analysis
- **Accuracy:**
  - Custom CNN Model: 89%
  - InceptionV3 Model: 98%
- **Error Analysis:**
  - Misclassified images highlighted challenges such as:
    - Variations in scan presentation (filling, orientation)
    - Darker tumor areas
    - Lateral/posterior views compared to superior (top-down) views
    ![Incorrect Predictions](https://github.com/kalebhings/Brain-Tumors-CNN/blob/main/incorrect%20images.png?raw=true)
- **Limitations:**
  - Potential overfitting to the training data
  - Applicability to diverse datasets (e.g., images from different MRI machines)

## Insights
- Leveraging transfer learning significantly improved performance, demonstrating its value in tasks with limited data availability.
- Data augmentation enhanced model robustness and generalization, but further exploration into task-specific augmentations could yield additional benefits.
- Misclassification patterns suggest a need for more diverse training data to account for variations in MRI scans.

## Future Work
- Explore additional pre-trained models to compare performance and robustness.
- Investigate techniques to mitigate overfitting, such as dropout or more advanced regularization methods.
- Expand the dataset to include images from diverse MRI machines and clinical conditions.
- Incorporate explainability methods (e.g., Grad-CAM) to visualize and understand the decision-making process of the model.

## Tools and Technologies
- **Programming Language:** Python
- **Frameworks:** TensorFlow, Keras
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn
- **Model:** InceptionV3 (Transfer Learning)
- **Dataset Source:** Kaggle
- **Development Environment:** Jupyter Notebook, Google Colab

---

**Author:** Kaleb Hingsberger
