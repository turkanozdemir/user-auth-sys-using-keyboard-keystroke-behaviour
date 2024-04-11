# Keylogger

## Overview

This module is designed to capture and analyze keystroke dynamics for user authentication based on unique typing patterns. The collected data is stored in a CSV file, and a separate script is provided for data analysis.

## Prerequisites
- [Ubuntu 22.04](https://ubuntu.com/download/desktop)
- [Python 3.10.x](https://www.python.org/)
- pip
- [Visual Studio Code](https://code.visualstudio.com/download) (optinal)

## Installation

1. **The script is specifically designed for Linux and is tested on [Ubuntu 22.04](https://ubuntu.com/download/desktop).**

2. **Ensure that you have [Python](https://www.python.org/) installed on your Ubuntu system.**

    ```bash
    python3 --version
    ```

3. **If `Python` is not installed, you can install Python using the following commands for Ubuntu:**

    ```bash
    sudo apt update
    sudo apt install python3
    ```

4. **Ensure that you have `pip` installed on your Ubuntu system.**

    ```bash
    pip --version
    ```

5. **If `pip` is not installed, you can install it by running:**

    ```bash
    sudo apt install python3-pip
    ```

6. **Install the required Python libraries by running:**

    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Running the Module

1. **Open a terminal in project folder in VS Code**
    ```bash
    cd keylogger
    ```

2. **To capture keystroke dynamics, run the main() function from the keylogger.py script. Follow the on-screen instructions, and press Escape when done to generate a CSV file:**

    ```bash
    python3 keylogger.py
    ```

3. **To analyze the captured keystroke data, run the main() function from the analyzer.py script:**

    ```bash
    python3 analyze.py
    ```

### Important Notes

- This application is designed to work on Ubuntu and may not be compatible with other operating systems.
- Follow on-screen instructions while capturing keystroke dynamics.
- Press Escape to finish capturing and generate the CSV file.
- Use the generated CSV file with the analyzer script to analysis users keystroke behaviour.
- If you encounter any issues or have suggestions, feel free to provide feedback. 