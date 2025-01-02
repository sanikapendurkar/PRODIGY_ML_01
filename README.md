# House Price Prediction Model

This project involves building a machine learning model to predict house prices based on various features. The dataset used includes attributes such as average area income, house age, number of rooms, number of bedrooms, and area population.

## About the Project

The objective of this project is to develop a linear regression model that accurately predicts house prices based on input features. The project includes data preprocessing, exploratory data analysis, model training, and evaluation.

## Dataset

The dataset used for this project is `USA_Housing.csv`, which contains the following columns:

- **Avg. Area Income**: Average income of residents in the area.
- **Avg. Area House Age**: Average age of houses in the area.
- **Avg. Area Number of Rooms**: Average number of rooms in houses in the area.
- **Avg. Area Number of Bedrooms**: Average number of bedrooms in houses in the area.
- **Area Population**: Population of the area.
- **Price**: Price at which the house is sold.
- **Address**: Address of the house.

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
- **Exploratory Data Analysis (EDA)**: Visualizing relationships between features and the target variable.
- **Model Training**: Implementing a linear regression model using the processed data.
- **Model Evaluation**: Assessing the model's performance using appropriate metrics.

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or JupyterLab
- Necessary Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/heemit/PRODIGY_ML_01.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd PRODIGY_ML_01
   ```
  
3. **Install the required packages**:
   ```bash
   cd pip install pandas numpy matplotlib seaborn scikit-learn
   ```

## Usage

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook Linear_Regression_Model.ipynb
   ```

2. **Run the cells sequentially:**
   Execute the cells in sequence to preprocess the data, train the model, and evaluate its performance.

## Model Evaluation

The model's performance is evaluated using the following metrics:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

Additionally, visualizations of the model's predictions versus actual values are provided to assess accuracy.
