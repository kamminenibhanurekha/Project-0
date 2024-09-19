import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,auc,roc_curve
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load the dataset
data = pd.read_csv("heart_attack_dataset.csv")

# Preview the data
print(data.head())

# Check for missing values and drop them if necessary
data = data.dropna()

# Convert categorical variables to numeric using LabelEncoder
categorical_columns = ['Gender', 'Has Diabetes', 'Smoking Status', 'Chest Pain Type']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Visualize the distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Visualize Blood Pressure by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Blood Pressure (mmHg)', data=data)
plt.title('Blood Pressure by Gender')
plt.show()

# Select only numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Calculate correlation matrix
corr_matrix = numeric_data.corr()

# Create correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Define features and target variable
X = data.drop(columns=['Treatment'])
y = data['Treatment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train,epochs=10)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


'''# Define the input data schema
class PatientData(BaseModel):
    gender: int
    age: int
    blood_pressure: int
    cholesterol: int
    has_diabetes: int
    smoking_status: int
    chest_pain_type: int

@app.post("/predict-treatment/")
def predict_treatment(data: PatientData):
    # Convert the input data to a NumPy array
    input_data = np.array([[data.gender, data.age, data.blood_pressure, data.cholesterol,
                            data.has_diabetes, data.smoking_status, data.chest_pain_type]])


    
    # Predict the treatment using the trained model
    prediction = model.predict(input_data)
    return {"predicted_treatment": int(prediction[0])}
@app.get("/predict/")

async def read_predict():

    raise HTTPException(status_code=405, detail="Method Not Allowed")


@app.put("/predict/apps")

async def update_predict():

    raise HTTPException(status_code=405, detail="Method Not Allowed")


@app.delete("/predict/")

async def delete_predict():

    raise HTTPException(status_code=405, detail="Method Not Allowed")




'''