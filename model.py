from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import os


def train_model(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    # =========================
    # SVM MODEL
    # =========================

    svm_model = SVC(kernel="linear", probability=True)

    svm_model.fit(X_train, Y_train)

    svm_predictions = svm_model.predict(X_test)

    svm_accuracy = accuracy_score(Y_test, svm_predictions)

    print("\nSVM Accuracy:", svm_accuracy * 100, "%")

    # Save SVM model
    os.makedirs("models", exist_ok=True)
    joblib.dump(svm_model, "models/watch_model.pkl")

    print("SVM model saved successfully!")


    # =========================
    # KNN MODEL
    # =========================

    knn_model = KNeighborsClassifier(n_neighbors=3)

    knn_model.fit(X_train, Y_train)

    knn_predictions = knn_model.predict(X_test)

    knn_accuracy = accuracy_score(Y_test, knn_predictions)

    print("KNN Accuracy:", knn_accuracy * 100, "%")


    # =========================
    # SAVE RESULTS
    # =========================

    os.makedirs("results", exist_ok=True)

    with open("results/accuracy.txt", "w") as file:

        file.write("Watch Detection Model Results\n")
        file.write("============================\n\n")

        file.write(
            "SVM Accuracy: "
            + str(svm_accuracy * 100)
            + "%\n"
        )

        file.write(
            "KNN Accuracy: "
            + str(knn_accuracy * 100)
            + "%\n"
        )


    # =========================
    # CONFUSION MATRIX
    # =========================

    cm = confusion_matrix(Y_test, svm_predictions)

    plt.figure()

    plt.imshow(cm)

    plt.title("SVM - Watch Detection Confusion Matrix")

    plt.xlabel("Predicted Label")

    plt.ylabel("Actual Label")

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Without Watch", "With Watch"]
    )

    plt.yticks(
        [0, 1],
        ["Without Watch", "With Watch"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.savefig(
        "results/confusion_matrix.png"
    )

    plt.close()

    print("Results saved successfully!")

    # Return SVM because it will be used by the application
    return svm_model
