import matplotlib.pyplot as plt
import numpy as np

from backend.visualization.utils import roc_curve_plots
from backend.model_scripts.utils import read_data
from backend.model_scripts import preprocessing


def roc_plot(data_path="data/titanic.csv"):

    X_train, _, y_train, _ = read_data(path=data_path)

    plt.figure(figsize=(8, 8))
    for i, clf in enumerate(preprocessing.classifiers, 1):
        plt.subplot(3, 2, i)
        class_name, fpr, tpr, _, roc_auc = roc_curve_plots(clf, X_train, y_train)
        plt.plot(fpr, tpr,  label = 'AUC = %0.2f' % roc_auc)

        distances = tpr - fpr
        idx = np.argmax(distances)
        elbow_x, elbow_y = fpr[idx], tpr[idx]

        plt.vlines(x=elbow_x, ymin=0, ymax=elbow_y, color='red', linestyle='--')
        plt.scatter(elbow_x, elbow_y, color='black', s=50)
        plt.text(elbow_x + 0.02, elbow_y - 0.05, f'FPR: {elbow_x:.3f}\nTPR: {elbow_y:.3f}\n', 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        plt.legend(loc = "lower right")
        plt.title(class_name)
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")

    plt.tight_layout(pad=3.0)
    plt.savefig("plots/roc_auc_plot")
    plt.show()

