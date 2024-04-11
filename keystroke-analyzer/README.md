# Keystroke Analyzer

## Overview

The Keystroke Analyzer is the user interface for the User Authentication Systems Using Keyboard Keystroke Behaviour project. With this interface, you can analyze datasets obtained from the Keylogger module and view the results on different models provided by Models backend application.

## Prerequisites

- [Node.js](https://nodejs.org/)
- [NPM](https://www.npmjs.com/get-npm) (Node Package Manager)
- [Visual Studio Code](https://code.visualstudio.com/download) (optinal)

## Installation

- Open a terminal or command prompt in the project directory.

    ```bash
    cd keystroke-analyzer
    ```

- Install project dependencies by running the following command:

    ```bash
    npm install react react-dom react-router-dom axios @mui/material @ag-grid-community/react papaparse
    ```

## Running the Keystroke Analyzer

- After the installation is complete successfully, start the Keystroke Analyzer by running:

    ```bash
    npm start
    ```

- This will launch the application, and you can access it in your web browser at [http://localhost:3000](http://localhost:3000).


## Usage

1. **File Upload:**

    - To upload your dataset in CSV format, you can use the "Upload Dataset" button.
    - Alternatively, you can drag and drop your dataset to upload it.
    - Ensure that your file has a .csv extension.

2. **Model Selection:**

    - Choose the machine learning model for analysis from the provided options in the dropdown menu:
         - Multi-Layered Perceptron (MLP), 
         - K-Nearest Neighbors (KNN), 
         - Support Vector Machine (SVM), 
         - Logistic Regression (LR), 
         - or select "All Models" for a combined analysis.    

3. **Split Ratio Selection:**

    - Select the desired split ratio for training and testing datasets from the provided options in the dropdown menu.

4. **Analysis:**

    - Click the "Analyze" button to start the analysis process.
    - The results will be displayed in a report, and you can navigate to the corresponding pages for detailed information.

### Important Notes

- Ensure that Models application is running for Keystroke Analyzer to analyze datasets successfully,
- Ensure that your dataset is in CSV format.
- The application uses different machine learning models for analysis, and you can choose the model that best suits your requirements.
- If you encounter any issues or have suggestions, feel free to provide feedback.