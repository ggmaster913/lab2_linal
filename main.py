import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from perceptron import Perceptron
from data_utils import standardize, stratified_split, generate_custom_data


# Функция расчета метрик из задания
def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1


def plot_decision_boundary(model, X, y, title="Разделяющая граница"):
    # Рисуем красивый график с прямой
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='bwr')

    # Отмечаем ошибки крестиками (желтыми)
    preds = model.predict(X)
    errors = X[y != preds]
    if len(errors) > 0:
        plt.scatter(errors[:, 0], errors[:, 1], marker='x', color='yellow', label='Ошибки')

    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 1. Готовим данные
    # X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
    #                            n_informative=2, random_state=42, n_clusters_per_class=1)

    # Будем юзать свой генератор для бонусов
    X, y = generate_custom_data(data_type='linear')
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train, X_test = standardize(X_train, X_test)

    # 2. Обучаем модель (базовые параметры из ТЗ)
    model = Perceptron(input_dim=2)
    train_loss, val_loss = model.fit(X_train, y_train, X_test, y_test,
                                     epochs=100, lr=0.1, batch_size=32, momentum_beta=0.9)

    # 3. Смотрим метрики
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train, _, _, _ = calculate_metrics(y_train, y_pred_train)
    acc_test, prec, rec, f1 = calculate_metrics(y_test, y_pred_test)

    print(f"Точность на трейне: {acc_train:.4f}")
    print(f"Точность на тесте: {acc_test:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # 4. Графики Loss
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.title('График падения ошибки')
    plt.legend()
    plt.show()

    # 5. Разделяющая граница
    plot_decision_boundary(model, X_test, y_test)