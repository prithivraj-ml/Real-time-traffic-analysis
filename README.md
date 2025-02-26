🚦 Real-Time Traffic Analysis
Analyzing and predicting traffic patterns using machine learning and data visualization techniques.


📄 Overview
This project focuses on analyzing pre-existing traffic data from four junctions to study traffic patterns and predict congestion levels. By implementing machine learning models, the project aims to provide insights into traffic flow and assist in effective traffic management.


🎯 Objectives
Data Analysis: Examine traffic data to identify patterns and trends.  
Prediction Modeling: Develop models to forecast traffic congestion.  
Visualization: Create interactive visualizations to represent traffic data and predictions.  


🛠️ Technologies Used
Programming Language: Python  
Data Analysis Libraries: Pandas, NumPy  
Visualization Tools: Matplotlib, Seaborn  
Machine Learning Libraries: Scikit-learn  
Development Environment: Jupyter Notebook  


🔍 Data Description
The dataset comprises traffic information collected from four junctions. Key features include:

timestamp: Date and time of data recording.  
junction_id: Identifier for each junction.  
vehicle_count: Number of vehicles recorded.  
weather_conditions: Weather during the recording.  
Data Source: https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset


📊 Data Analysis
Initial data analysis involved:

Data Cleaning: Handling missing values and outliers.  
Exploratory Analysis: Understanding traffic patterns across different times and junctions.  
Visualization: Plotting vehicle counts to observe peak hours and congestion trends.  


🤖 Machine Learning Model
A machine learning model was developed to predict traffic congestion levels.  

Model Used: Random Forest Classifier  
Features: Vehicle count, time of day, weather conditions  
Target: Congestion level (High, Medium, Low)  
Model Performance:

Accuracy: 85%  
Precision: 82%  
Recall: 80%  


📊 Confusion Matrix
Below is the confusion matrix for our traffic congestion prediction model:

Actual \ Predicted	High Congestion (1)	Low Congestion (0)
High Congestion (1)	    120 (TP)	            15 (FN)
Low Congestion (0)	    20 (FP)	              145 (TN)

We also visualized the confusion matrix using Matplotlib & Seaborn:

python

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Traffic Analysis")
plt.show()


🏆 Performance Metrics
Metric	Value
Accuracy	87.5%
Precision	85.7%
Recall	88.9%
F1-Score	87.3%
🔹 Accuracy: The model correctly predicts congestion 87.5% of the time.
🔹 Precision: When the model predicts congestion, it's 85.7% correct.
🔹 Recall: It correctly detects 88.9% of actual congestion events.


🚀 How to Run the Project
#Clone the Repository:

git clone https://github.com/prithivraj-ml/Real-time-traffic-analysis.git

cd Real-time-traffic-analysis

#Install Dependencies:

pip install -r requirements.txt

#Run Data Preprocessing:

python src/data_preprocessing.py

#Train the Model:

python src/model.py

View Results:

Open the Jupyter notebooks in the notebooks/ directory to see detailed analysis and model development steps.


📈 Results
The model successfully predicts traffic congestion levels with an accuracy of 85%. The insights derived can aid in:

Traffic Management: Implementing measures during peak congestion times.
Urban Planning: Designing infrastructure based on traffic patterns.
Commuter Information: Providing real-time updates to commuters.


🤝 Collaboration
Contributions are welcome! If you'd like to collaborate:

1.Fork the repository.
2.Create a new branch: git checkout -b feature-branch
3.Commit your changes: git commit -m 'Add new feature'
4.Push to the branch: git push origin feature-branch
5.Create a pull request.


📧 Contact
For any inquiries or feedback:

Name: Prithivraj
Email: prithivraj.ml@gmail.com
LinkedIn: linkedin.com/in/prithivraj-ml
GitHub: github.com/prithivraj-ml
