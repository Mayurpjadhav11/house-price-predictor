# 🏠 House Price Predictor

An end-to-end machine learning project that predicts median house values using the **California Housing dataset**.

This project demonstrates the complete workflow of building a production-ready machine learning application:

```text
Data → Preprocessing → Model Training → Evaluation
     → Model Saving → FastAPI → Docker → Testing → Deployment
```

The project begins with a **Linear Regression** baseline and then uses a **Random Forest Regressor** to improve prediction performance.

---

## 📌 Project Overview

Predicting house prices is a common regression problem in machine learning.

The objective of this project is to train a model that learns the relationship between housing-related features and the median house value in a particular California district.

The project is not limited to training a model inside a notebook. It also demonstrates how to:

* organize a machine learning repository,
* create a reusable training pipeline,
* save and load a trained model,
* expose predictions through a REST API,
* validate user input,
* test the application,
* package the application using Docker,
* automate testing using GitHub Actions,
* and deploy the model as an online service.

---

## 🎯 Project Objective

The main objective is to predict the median house value using information such as:

* median income,
* average house age,
* average number of rooms,
* average number of bedrooms,
* population,
* average household occupancy,
* latitude,
* longitude.

This is a **supervised regression problem** because:

* the dataset contains input features,
* the correct house value is available during training,
* and the model predicts a continuous numerical value.

---

## 🧠 Machine Learning Workflow

The complete workflow followed in this project is:

```text
1. Load the California Housing dataset
2. Explore and understand the data
3. Separate input features and target
4. Split the data into training and testing sets
5. Train a Linear Regression baseline
6. Train a Random Forest Regressor
7. Compare model performance
8. Select the best-performing model
9. Save the trained model using Joblib
10. Load the model inside a FastAPI application
11. Accept house information through an API
12. Return the predicted house value
13. Test and containerize the application
```

---

## 📊 Dataset

This project uses the **California Housing dataset** available through Scikit-learn.

The dataset contains information collected from different California housing districts.

### Input features

| Feature      | Description                              |
| ------------ | ---------------------------------------- |
| `MedInc`     | Median income in the district            |
| `HouseAge`   | Median age of houses in the district     |
| `AveRooms`   | Average number of rooms per household    |
| `AveBedrms`  | Average number of bedrooms per household |
| `Population` | Population of the district               |
| `AveOccup`   | Average number of people per household   |
| `Latitude`   | Geographical latitude                    |
| `Longitude`  | Geographical longitude                   |

### Target variable

| Target        | Description                         |
| ------------- | ----------------------------------- |
| `MedHouseVal` | Median house value for the district |

The dataset can be loaded directly using Scikit-learn:

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame
```

---

## 🏗️ Project Structure

```text
house-price-predictor/
│
├── app/
│   └── main.py
│
├── data/
│   └── README.md
│
├── models/
│   └── house_price_model.joblib
│
├── notebooks/
│   └── model_experiment.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

### Folder explanation

| Folder/File          | Purpose                                         |
| -------------------- | ----------------------------------------------- |
| `app/`               | Contains the FastAPI application                |
| `data/`              | Stores local dataset-related files              |
| `models/`            | Stores the trained machine learning model       |
| `notebooks/`         | Contains experiments and exploratory analysis   |
| `src/`               | Contains training and prediction scripts        |
| `tests/`             | Contains automated tests                        |
| `.github/workflows/` | Contains the GitHub Actions CI workflow         |
| `Dockerfile`         | Defines the Docker container                    |
| `requirements.txt`   | Lists required Python packages                  |
| `.gitignore`         | Prevents unnecessary files from being committed |
| `LICENSE`            | Defines how the project may be reused           |

---

## 🤖 Models

Two machine learning models are compared in this project.

### 1. Linear Regression

Linear Regression is used as the baseline model.

It attempts to learn a linear relationship between the housing features and the target value.

Advantages:

* simple,
* fast to train,
* easy to interpret,
* useful as a baseline.

Limitations:

* assumes mostly linear relationships,
* may not capture complex patterns,
* can be sensitive to unusual values.

### 2. Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

Advantages:

* captures nonlinear relationships,
* handles feature interactions,
* usually performs well on tabular data,
* requires less feature scaling,
* is more flexible than Linear Regression.

Limitations:

* larger model size,
* slower than Linear Regression,
* less directly interpretable.

---

## 📏 Evaluation Metrics

The models are evaluated using the following regression metrics.

### Mean Absolute Error — MAE

MAE calculates the average absolute difference between the actual and predicted values.

```text
Lower MAE = Better model
```

MAE is easy to understand because it measures the average prediction error.

### Root Mean Squared Error — RMSE

RMSE gives more importance to large errors because the errors are squared before averaging.

```text
Lower RMSE = Better model
```

RMSE is useful when large prediction mistakes should be penalized more strongly.

### R² Score

The R² score measures how much of the target variation is explained by the model.

```text
R² close to 1 = Strong model performance
R² close to 0 = Weak explanatory performance
R² below 0    = Worse than predicting the average
```

---

## 📈 Model Results

The final results will be added after training both models.

| Model             |         MAE |        RMSE |          R² |
| ----------------- | ----------: | ----------: | ----------: |
| Linear Regression | To be added | To be added | To be added |
| Random Forest     | To be added | To be added | To be added |

The model with the best testing performance will be saved and used by the FastAPI application.

> Important: The testing dataset must not be used during model training.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* FastAPI
* Pydantic
* Uvicorn
* Pytest
* Docker
* Git and GitHub
* GitHub Actions

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
```

Move into the project folder:

```bash
cd house-price-predictor
```

Replace `YOUR_USERNAME` with your GitHub username.

---

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

#### macOS or Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚂 Train the Model

Run the training script from the project root:

```bash
python src/train.py
```

The training script will:

1. load the dataset,
2. split the data,
3. train the models,
4. evaluate their performance,
5. select the best model,
6. save the selected model.

The trained model will be stored at:

```text
models/house_price_model.joblib
```

---

## 🔮 Test the Saved Model

After training, test the saved model using:

```bash
python src/predict.py
```

This script loads the model and makes a prediction using example housing values.

This confirms that the saved model can be loaded correctly before it is connected to the API.

---

## 🚀 Run the FastAPI Application

Start the API using:

```bash
uvicorn app.main:app --reload
```

The application will run at:

```text
http://127.0.0.1:8000
```

Open the automatic API documentation:

```text
http://127.0.0.1:8000/docs
```

The `/docs` page allows you to test the prediction endpoint directly from the browser.

---

## 🔌 API Endpoints

### Home endpoint

```http
GET /
```

Example response:

```json
{
  "message": "House Price Predictor API is running"
}
```

---

### Health endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

The health endpoint confirms that the application is running.

---

### Prediction endpoint

```http
POST /predict
```

Example request:

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

Example response:

```json
{
  "predicted_house_value": 4.52
}
```

The exact value depends on the trained model.

---

## ✅ Input Validation

The API uses Pydantic models to validate incoming data.

Validation helps prevent incorrect input such as:

```json
{
  "median_income": "unknown",
  "house_age": -50
}
```

The API checks whether:

* required fields are present,
* numerical values use the correct data type,
* values follow defined limits,
* invalid requests are rejected before reaching the model.

---

## 🧪 Run Tests

Run the automated tests using:

```bash
pytest
```

The tests verify that:

* the application starts,
* the health endpoint works,
* the prediction endpoint accepts valid input,
* a prediction is returned,
* invalid input is rejected correctly.

---

## 🐳 Docker

Docker packages the application, dependencies, and trained model into a portable container.

### Build the Docker image

```bash
docker build -t house-price-predictor .
```

### Run the Docker container

```bash
docker run -p 8000:8000 house-price-predictor
```

Open:

```text
http://localhost:8000/docs
```

The application should behave the same way inside Docker as it does in the local Python environment.

---

## 🔄 Continuous Integration

GitHub Actions will automatically run the project tests whenever code is pushed to the repository.

The CI workflow performs steps such as:

```text
Push code to GitHub
        ↓
Create Python environment
        ↓
Install dependencies
        ↓
Run automated tests
        ↓
Report pass or failure
```

This helps detect problems before changes are deployed.

---

## ☁️ Deployment

The API can later be deployed using a cloud platform that supports Python or Docker applications.

The deployed application will allow users or other applications to send house information and receive predictions through an online API.

Deployment steps will be added after the local API, tests, and Docker container are working correctly.

---

## 🗺️ Development Roadmap

* [x] Create GitHub repository
* [x] Add README, `.gitignore`, and MIT License
* [ ] Create local project structure
* [ ] Set up Python virtual environment
* [ ] Create exploratory notebook
* [ ] Train Linear Regression model
* [ ] Train Random Forest model
* [ ] Compare model performance
* [ ] Save the best model
* [ ] Create local prediction script
* [ ] Build FastAPI application
* [ ] Add input validation
* [ ] Add automated tests
* [ ] Create Docker image
* [ ] Add GitHub Actions
* [ ] Deploy the application
* [ ] Add final model metrics
* [ ] Add API screenshots

---

## 💡 What This Project Demonstrates

This repository demonstrates practical knowledge of:

* supervised machine learning,
* regression modelling,
* model comparison,
* data preprocessing,
* model serialization,
* REST API development,
* API input validation,
* software testing,
* containerization,
* version control,
* continuous integration,
* machine learning deployment.

---

## ⚠️ Limitations

* The model is trained on historical California housing data.
* Predictions should not be treated as professional property valuations.
* The model may not generalize to houses outside the dataset’s geographical and historical context.
* Prediction quality depends on data quality and feature availability.
* The project is intended for educational and portfolio purposes.

---

## 🔮 Future Improvements

Possible future improvements include:

* hyperparameter tuning,
* cross-validation,
* additional regression algorithms,
* feature importance analysis,
* SHAP explainability,
* model monitoring,
* experiment tracking,
* data validation,
* a web-based user interface,
* automatic cloud deployment,
* retraining pipelines.

---

## 📄 License

This project is licensed under the MIT License.

You may use, modify, and distribute the code while preserving the original license notice.

See the `LICENSE` file for complete details.

---

## 👤 Author

**Your Name**

Machine Learning and AI Engineering Enthusiast

GitHub: `https://github.com/YOUR_USERNAME`

---

## ⭐ Support

If this project is useful, consider giving the repository a star.

Contributions, suggestions, and improvements are welcome.
****
