# PolitifactClassifier: Predicting the Truthfulness of Politician Statements

## Overview

The PolitifactClassifier project is a binary classification task aimed at predicting the truthfulness of statements made by politicians. The project leverages multiple machine learning approaches, including a scikit-learn solution, a deep learning solution, and an ensemble solution, to achieve high accuracy and robustness in classification.

## Installation

To get started with the PolitifactClassifier, follow these installation steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/spkgyk/PolitifactClassifier.git
    cd PolitifactClassifier
    ```

2. **Set up the environment**:
    ```bash
    source setup/setup.sh
    ```

## Usage

### Running the Classifier

1. **Open the Jupyter Notebook**:
    Navigate to the `flip.ipynb` file and open it using Jupyter Notebook:
    ```bash
    jupyter notebook flip.ipynb
    ```

2. **Execute the Notebook**:
    Run all cells in the notebook to see the results of the following solutions:
    - **Scikit-learn Solution**: A traditional machine learning approach using scikit-learn.
    - **Deep Learning Solution**: An advanced neural network model for classification.
    - **Ensemble Solution**: A combination of multiple models to improve prediction accuracy (does not actually improve results lol).

## Project Structure

The repository is structured as follows:

- `setup/`: Contains the setup script for environment configuration.
- `data/`: Includes datasets and data processing scripts.
- `models/`: Houses the machine learning and deep learning models.
- `src/`: Source code for the classifier implementation.
- `flip.ipynb`: Main notebook for running and evaluating the classifier.

## Results and Evaluation

Detailed results and evaluation metrics for each solution can be found within the `flip.ipynb` notebook. This includes accuracy, precision, recall, and F1 scores, along with visualizations of the model performances.

## Contributing

We welcome contributions to enhance the PolitifactClassifier. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or inquiries, please contact [Your Name](mailto:spkgyk@outlook.com).

