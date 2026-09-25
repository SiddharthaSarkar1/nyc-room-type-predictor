# NYC Airbnb Room Type Classification

This project is an end-to-end machine learning solution that predicts the room type of an Airbnb listing in New York City (`Entire home/apt`, `Private room`, or `Shared room`) based on various listing attributes. It covers the complete ML lifecycle, from data exploration and cleaning to model comparison, tuning, and deployment via a REST API.

## Key Features

-   **Comprehensive Data Analysis:** Includes thorough Exploratory Data Analysis (EDA) covering univariate, bivariate, and correlation analysis.
-   **Robust Preprocessing:** Implements a scikit-learn `ColumnTransformer` pipeline to handle missing values, scaling, and one-hot encoding without data leakage.
-   **Multi-Model Comparison:** Evaluates and compares four different classification algorithms: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
-   **Hyperparameter Tuning:** Uses `RandomizedSearchCV` to find the optimal hyperparameters for the best-performing model (Random Forest), optimizing for macro F1-score to handle class imbalance.
-   **Final Evaluation:** Provides an honest performance assessment on a held-out test set, including accuracy, F1-score, and a confusion matrix.
-   **RESTful API:** Deploys the final tuned model as a production-ready API using FastAPI, allowing for real-time predictions.
-   **Dockerized:** Includes a `Dockerfile` for easy and reproducible deployment.

## Dataset

The project uses the **New York City Airbnb Open Data** dataset from Kaggle. This dataset includes various attributes for each listing, such as:

-   `price`
-   `minimum_nights`
-   `neighbourhood_group` and `neighbourhood`
-   `latitude` and `longitude`
-   `number_of_reviews` and `reviews_per_month`
-   `availability_365`

## Machine Learning Workflow

The project follows a structured machine learning lifecycle:

1.  **Data Cleaning & Feature Engineering:**
    -   Irrelevant columns (`id`, `name`, `host_id`, etc.) are dropped.
    -   Missing values in `reviews_per_month` are imputed with 0.
    -   Extreme outliers in `price` and `minimum_nights` are capped at the 99th percentile to prevent model distortion.

2.  **Train/Test Split:**
    -   The data is split into training (67%) and testing (33%) sets.
    -   The split is stratified on the target variable (`room_type`) to maintain class proportions.

3.  **Model Comparison:**
    -   Four models are evaluated using 3-fold cross-validation.
    -   **Random Forest** is selected as the best-performing model based on its high accuracy and F1-score.

4.  **Hyperparameter Tuning:**
    -   The Random Forest model is tuned using `RandomizedSearchCV` to find the best combination of `n_estimators`, `max_depth`, and `min_samples_split`.

5.  **Final Evaluation:**
    -   The final tuned pipeline achieves an **accuracy of 85.5%** and a **macro F1-score of 0.735** on the unseen test set.

## How to Use the API

The final model is served via a FastAPI application.

### 1. Setup

**Using Docker (Recommended):**

1.  **Build the Docker image:**
    ```bash
    docker build -t nyc-airbnb-predictor .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 nyc-airbnb-predictor
    ```

**Running Locally:**

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

The API will be available at `http://localhost:8000`.

### 2. API Endpoints

-   `GET /health`: Health check endpoint to confirm the server is running.
-   `GET /`: Welcome message.
-   `POST /predict`: Predicts the room type for a given set of features.

**Example `POST /predict` Request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "latitude": 40.75,
  "longitude": -73.98,
  "price": 150,
  "minimum_nights": 2,
  "number_of_reviews": 10,
  "reviews_per_month": 0.5,
  "calculated_host_listings_count": 1,
  "availability_365": 200,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}'
