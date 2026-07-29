# 🏠 House Price Predictor

An end-to-end machine learning project that predicts median house values in California using geographical, demographic, and housing-related information.

This project demonstrates the complete machine learning workflow—from data exploration and model training to API development, Dockerization, automated testing, and cloud deployment.

---

## Project Overview

The goal of this project is to build a reliable regression model that estimates house prices using the **California Housing dataset**.

Two machine learning models are compared:

* **Linear Regression** — used as a simple baseline model
* **Random Forest Regressor** — used to capture more complex relationships in the data

The best-performing model is saved and served through a **FastAPI REST API**, allowing other applications to request predictions.

---

## Machine Learning Workflow

```text
Dataset
   ↓
Data exploration and preprocessing
   ↓
Train Linear Regression
   ↓
Train Random Forest
   ↓
Evaluate and compare models
   ↓
Save the best model
   ↓
Serve predictions with FastAPI
   ↓
Package the application with Docker
   ↓
Test and deploy
```

---

## Dataset

The project uses the California Housing dataset available through Scikit-learn.

The dataset contains information such as:

* Median household income
* Average house age
* Average number of rooms
* Average number of bedrooms
* Population
* Average household occupancy
* Latitude
* Longitude

The target variable is the median house value for each California district.

---

## Model Evaluation

The models are evaluated using:

| Metric   | Meaning                                          |
| -------- | ------------------------------------------------ |
| MAE      | Average absolute prediction error                |
| RMSE     | Gives more importance to large prediction errors |
| R² Score | Measures how much variation the model explains   |

The model with the best overall performance is saved using `joblib`.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* FastAPI
* Pydantic
* Uvicorn
* Pytest
* Docker
* GitHub Actions

---

## Project Structure

```text
house-price-predictor/
│
├── app/
│   └── main.py              # FastAPI application
│
├── data/
│   └── README.md            # Dataset information
│
├── models/
│   └── house_price_model.joblib
│
├── notebooks/
│   └── model_experiment.ipynb
│
├── src/
│   ├── train.py             # Model training
│   └── predict.py           # Local prediction test
│
├── tests/
│   └── test_api.py          # API tests
│
├── .github/
│   └── workflows/
│       └── ci.yml           # Automated testing
│
├── .gitignore
├── Dockerfile
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mayurjadhav11/house-price-predictor.git
cd house-price-predictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```


Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run the training script:

```bash
python src/train.py
```



---

## Run the FastAPI Application

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Prediction Endpoint

### Endpoint

```text
POST /predict
```

### Example request

```json
{
  "median_income": 8.3,
  "house_age": 41,
  "average_rooms": 6.9,
  "average_bedrooms": 1.0,
  "population": 320,
  "average_occupancy": 2.5,
  "latitude": 37.88,
  "longitude": -122.23
}
```

### Example response

```json
{
  "predicted_house_value": 4.52
}
```

---

## Run Tests

```bash
pytest
```

The tests verify that:

* The API is running correctly
* The health endpoint responds successfully
* The prediction endpoint accepts valid input
* The model returns a prediction

---

## Run with Docker

Build the Docker image:

```bash
docker build -t house-price-predictor .
```

Run the container:

```bash
docker run -p 8000:8000 house-price-predictor
```

Open:

```text
http://localhost:8000/docs
```

---

## Future Improvements

* Add advanced hyperparameter tuning
* Track experiments using MLflow
* Add model explainability using SHAP
* Create a simple web interface
* Add data and model validation
* Deploy the API to a cloud platform
* Monitor model performance after deployment

---

## Learning Outcomes

This project demonstrates practical knowledge of:

* Regression modelling
* Model comparison and evaluation
* Saving and loading trained models
* Building REST APIs for machine learning
* Input validation
* Automated testing
* Docker containerization
* Git and GitHub workflow
* CI/CD fundamentals
* Cloud deployment

---

## License

This project is licensed under the MIT License.

You may use, modify, and distribute the code while keeping the original license notice.

---

## Author

**Mayur Jadhav**

Master’s student in AI Engineering of Autonomous Systems, interested in machine learning, autonomous systems, simulation, and production-ready AI applications.
