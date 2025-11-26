import matplotlib.pyplot as plt
import seaborn as sns

from backend.model_scripts.utils import read_data
from backend.model_scripts import preprocessing
from backend.visualization.utils import conf_matrix


def confusion_plot(data_path="data/titanic.csv"):

    X_train, _, y_train, _ = read_data(path=data_path)

    plt.figure(figsize=(15,8))
    for i , clf in enumerate(preprocessing.classifiers, 1):
        class_name, conf_val = conf_matrix(clf, X_train, y_train)
        plt.subplot(2, 3, i)
        sns.heatmap(conf_val, annot=True, 
                    fmt="d", cbar=False,  
                    cmap="Blues")
        plt.title(class_name)
        plt.xlabel("predicted label")
        plt.ylabel("True label")

    plt.tight_layout()

    plt.savefig("plots/confusion_matrix.png")
    plt.show()

# if __name__ == "__main__":
#     confusion_plot()