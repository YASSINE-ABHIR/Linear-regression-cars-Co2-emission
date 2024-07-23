
# Linear Regression on Cars' CO2 Emissions

This project demonstrates the use of linear regression to predict CO2 emissions from cars based on various features like engine size, number of cylinders, and fuel consumption.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you will need Python and the following libraries:
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using pip:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Usage

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/YASSINE-ABHIR/Linear-regression-cars-Co2-emission.git
cd Linear-regression-cars-Co2-emission
```

Run the script to train the model and generate visualizations:
```bash
python main.py
```

## Data

The dataset used in this project contains information about cars' engine size, number of cylinders, combined fuel consumption, and CO2 emissions. The data is loaded from a CSV file:

```python
df = pd.read_csv("./cars_CO2_emission.csv")
```

## Model Training

The features selected for training are:
- `ENGINESIZE`
- `CYLINDERS`
- `FUELCONSUMPTION_COMB`

The target variable is:
- `CO2EMISSIONS`

The dataset is split into training and test sets, and a linear regression model is trained:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train, test = train_test_split(data, test_size=0.2, random_state=42)
x_train = train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_train = train['CO2EMISSIONS']
x_test = test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_test = test['CO2EMISSIONS']

model = LinearRegression()
model.fit(x_train, y_train)
```

## Evaluation

The model's performance is evaluated using the mean squared error and the R^2 score:

```python
from sklearn.metrics import mean_squared_error, r2_score

predictions = model.predict(x_test)
print('Coefficients:', model.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))
print('Variance score: %.2f' % r2_score(y_test, predictions))
```

## Visualizations

### Distribution of CO2 Emissions
```python
plt.figure(figsize=(10, 6))
plt.hist(data['CO2EMISSIONS'], bins=30, color="lightblue", ec="black")
plt.xlabel("CO2 Emissions")
plt.ylabel("Frequency")
plt.title("Distribution of CO2 Emissions")
plt.show()
```
![Distribution of CO2 Emissions](https://github.com/YASSINE-ABHIR/Linear-regression-cars-Co2-emission/assets/60442896/1fd2646d-5cf3-4970-adcb-605360031e17)

### Box Plot of CO2 Emissions
```python
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['CO2EMISSIONS'], color='lightgreen')
plt.xlabel("CO2 Emissions")
plt.title("Box Plot of CO2 Emissions")
plt.show()
```
![Box Plot of CO2 Emissions](https://github.com/YASSINE-ABHIR/Linear-regression-cars-Co2-emission/assets/60442896/68b6bda6-824c-4b0f-8296-c588bbf424e9)

### Pair Plot of Features and CO2 Emissions
```python
sns.pairplot(data, diag_kind='kde', markers='+')
plt.suptitle("Pair Plot of Features and CO2 Emissions", y=1.02)
plt.show()
```
![Pair Plot of Features and CO2 Emissions](https://github.com/YASSINE-ABHIR/Linear-regression-cars-Co2-emission/assets/60442896/072e411d-6de4-4a68-82e4-7e488614aae7)

### Residual Plot
```python
plt.figure(figsize=(10, 6))
sns.residplot(x=predictions, y=y_test - predictions, lowess=True, color='hotpink')
plt.xlabel("Predicted CO2 Emissions")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
```
![Residual Plot](https://github.com/YASSINE-ABHIR/Linear-regression-cars-Co2-emission/assets/60442896/ce75ed46-c347-4cd2-9f16-b9f8a03111a7)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

