# Models

## Overview
This Django application serves as the backend for a Keystroke Analyzer User Interface (UI). The backend is designed to analyze keystroke data using machine learning classification models. It supports models such as Support Vector Machine (SVM), Multi-Layer Perceptron (MLP), K-Nearest Neighbors (KNN), and Logistic Regression (LR). The application provides various visualizations, including confusion matrices, correlation matrices, feature importance plots, ROC curves, and an accuracy comparison bar chart for all models.

### Features
- **SVM Predictions Endpoint**: Analyze keystroke data using a Support Vector Machine model.
- **MLP Predictions Endpoint**: Analyze keystroke data using a Multi-Layer Perceptron model.
- **KNN Predictions Endpoint**: Analyze keystroke data using a K-Nearest Neighbors model.
- **LR Predictions Endpoint**: Analyze keystroke data using a Logistic Regression model.
- **All Models Endpoint**: Compare the accuracy of all models on the given dataset.
- **Visualizations**: Generate visualizations such as confusion matrices, correlation matrices, feature importance plots, ROC curves, and an accuracy comparison bar chart.

## Prerequisites
- [Python 3.10.x](https://www.python.org/)
- Django
- [Visual Studio Code](https://code.visualstudio.com/download) (optinal)


## Installation
1. **Open a terminal and navigate to the project directory.** 
    ```bash
    cd models
    ```

2. **Install the required Python libraries by running:**

    ```bash
    pip install -r requirements.txt
    ```
   
## Running the Models 

- After the installation is complete successfully:

1. **Apply migrations**
    ```bash
    python manage.py migrate
    ```
   
2. **Run the development server**
    ```bash
    python manage.py runserver
    ```
After running these commands, the Models should be accessible at http://localhost:8000 

### Important Notes

- Use the Keystroke Analyser UI to use the application effectively.

Feel free to customize the scripts or incorporate them into your authentication system. If you encounter any issues or have suggestions, please provide feedback or report them to the repository. 

Thank you for using Models!