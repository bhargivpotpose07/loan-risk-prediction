import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

print(df.head())

# Fill missing values

for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

# Split data
from sklearn.model_selection import train_test_split

X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Graph
sns.countplot(x="Loan_Status", data=pd.read_csv("train.csv"))
plt.title("Loan Status Distribution")
plt.show()