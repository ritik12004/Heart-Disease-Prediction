❤️ Heart Disease Prediction

The Heart Disease Prediction project is a Machine Learning–based application developed to predict the likelihood of heart disease using patient health parameters. The project demonstrates the complete end-to-end ML workflow, from data preprocessing and model training to saving trained models and making predictions through a simple user interface.

This system aims to showcase how machine learning can assist in early risk assessment and decision support in the healthcare domain.

🎯 Objectives

To build a predictive model for heart disease detection

To apply real-world machine learning techniques on healthcare data

To understand feature scaling, model evaluation, and serialization

To deploy a trained model for real-time predictions

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Streamlit

Pickle

GitHub

⚙️ Key Features

Data cleaning and preprocessing

Exploratory Data Analysis (EDA) and visualization

Feature scaling using StandardScaler

Model training and evaluation

Saved trained model (model.pkl) and scaler (scaler.pkl)

User interface for real-time prediction

🧠 System Design

heart.csv contains the dataset used for training and evaluation

heart (1).ipynb includes data analysis, preprocessing, and model training

model.pkl stores the trained machine learning model

scaler.pkl stores the fitted feature scaler

ui.py provides a simple interface for user input and prediction

requirements.txt lists all required dependencies

📂 Project Structure
Heart-Disease-Prediction/
│
├── heart.csv                 # Dataset
├── heart (1).ipynb           # Data analysis & model training notebook
├── model.pkl                 # Trained ML model
├── scaler.pkl                # Feature scaler
├── ui.py                     # User interface for prediction
├── requirements.txt          # Project dependencies
├── LICENSE                   # MIT License
└── README.md

▶️ How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/ritik12004/Heart-Disease-Prediction.git

2️⃣ Navigate to the project directory
cd Heart-Disease-Prediction

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
streamlit run ui.py

📊 Learning Outcomes

Hands-on experience with healthcare datasets

Strong understanding of ML preprocessing and feature scaling

Practical exposure to model evaluation and deployment

Experience in saving and loading trained models

🚀 Future Enhancements

Improve model performance using advanced algorithms

Handle class imbalance more effectively

Add explainability using SHAP or LIME

Deploy the application on cloud platforms

👤 Author

Ritik Gujre
Student | Aspiring Data Scientist | Python Developer
Samrat Ashok Technological Institute, Vidisha

GitHub: https://github.com/ritik12004

LinkedIn: https://www.linkedin.com/in/ritikgujre/

Email: ritik26cs103@satiengg.in

📜 License

This project is licensed under the MIT License.
See the LICENSE file for more details.
