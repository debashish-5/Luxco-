import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


#load data
df = pd.read_csv('professional_crop_medicine_dataset_6000_rows.csv')

#model loading
onehot = OneHotEncoder(handle_unknown='ignore')
data = onehot.fit_transform(df[['Crop','Problem', 'Weather', 'Soil_Type', 'Growth_Stage', 'Severity']])
tree_model = DecisionTreeClassifier()
tree_model.fit(data,df["Recommended_Medicine"])
print("Tree model Trained")
#prediction function
def predict_medicine(input_data):
    encode = onehot.transform(input_data)
    prediction = tree_model.predict(encode)[0]
    return prediction

