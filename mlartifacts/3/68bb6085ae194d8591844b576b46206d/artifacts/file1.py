import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)


# Define parameters for Random Forest
max_depth = 10
n_estimators = 5


# Create/select MLflow experiment
# Exp1 was deleted, so we use a new experiment name
mlflow.set_experiment("YT-MLOPS-Exp2")


# Start MLflow run
with mlflow.start_run():

    # Create Random Forest model
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    # Train model
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)


    # Log parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)


    # Log metric
    mlflow.log_metric("accuracy", accuracy)


    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))

    # Display confusion matrix using Matplotlib
    plt.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.colorbar()

    # Class labels
    tick_marks = range(len(wine.target_names))

    plt.xticks(
        tick_marks,
        wine.target_names
    )

    plt.yticks(
        tick_marks,
        wine.target_names
    )


    # Add values inside confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )


    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()


    # Save confusion matrix
    plt.savefig("Confusion-matrix.png")

    # Close plot
    plt.close()


    # Log confusion matrix as artifact
    mlflow.log_artifact("Confusion-matrix.png")

    # Log source code as artifact
    mlflow.log_artifact(__file__)


    # Add tags
    mlflow.set_tags({
        "Author": "SAMEER",
        "Project": "Wine Classification"
    })


    # Log Random Forest model
    mlflow.sklearn.log_model(
        rf,
        "Random-Forest-Model"
    )


    # Print accuracy
    print("Accuracy:", accuracy)
