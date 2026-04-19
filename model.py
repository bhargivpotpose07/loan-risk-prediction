import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():

    # Load dataset
    df = pd.read_csv("train.csv")

    # Drop ID column
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # Drop missing values
    df = df.dropna()

    # Encode target
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    # Convert categorical variables to numeric
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Return model + column structure
    return model, X.columns