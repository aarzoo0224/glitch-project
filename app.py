from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv('car1price.csv',encoding='utf-8')
X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']

# Encode the categorical values
le = LabelEncoder()
X['Fuel_Type'] = le.fit_transform(X['Fuel_Type'])
X['Seller_Type'] = le.fit_transform(X['Seller_Type'])
X['Transmission'] = le.fit_transform(X['Transmission'])

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

# loading the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/project')
def project():
    return render_template("form.html")

@app.route('/predict',methods=['POST'])
def predict():
    Year = int(request.form['Year'])
    Present_Price = float(request.form['Present_Price'])
    Kms_Driven = int(request.form['Kms_Driven'])

    Fuel_Type = request.form['Fuel_Type']
    Seller_Type = request.form['Seller_Type']
    Transmission = request.form['Transmission']
    Owner = int(request.form['Owner'])

    # Encode the categorical values
    Fuel_Type = le.transform([Fuel_Type])[0]
    Seller_Type = le.transform([Seller_Type])[0]
    Transmission = le.transform([Transmission])[0]

    # Create a new row with the user input
    new_row = {'Year': Year, 'Present_Price': Present_Price, 'Kms_Driven': Kms_Driven, 'Fuel_Type': Fuel_Type, 'Seller_Type': Seller_Type, 'Transmission': Transmission, 'Owner': Owner}

    # Convert the new row to a DataFrame
    new_row_df = pd.DataFrame(new_row, index=[0])

    # Predict the selling price of the car
    predicted_price = lin_reg_model.predict(new_row_df)[0]

    # Print the predicted selling price
    print("The predicted selling price of the car is: " + str(predicted_price))

    return render_template("form.html")

if __name__ == "__main__":
    app.run()

