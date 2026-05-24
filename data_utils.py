import numpy as np


def generate_custom_data(n_samples=500, data_type='linear', noise_prob=0.05):
    # Бонусное задание: свой генератор
    np.random.seed(42)

    if data_type == 'linear':
        # Два гауссовых облака
        X0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
        X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
        X = np.vstack((X0, X1))
        y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    elif data_type == 'xor':
        # Нелинейные данные (крест-накрест)
        X = np.random.randn(n_samples, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

    elif data_type == 'circle':
        # Точки внутри и снаружи круга
        X = np.random.randn(n_samples, 2) * 2
        radius = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        y = (radius > 2.0).astype(int)

    # Добавляем шум (инвертируем часть меток)
    if noise_prob > 0:
        flip_mask = np.random.rand(n_samples) < noise_prob
        y[flip_mask] = 1 - y[flip_mask]

    return X, y


def standardize(X_train, X_test):
    # Z-нормализация: вычитаем среднее, делим на отклонение
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)

    # Чтобы не поделить на ноль случайно
    sigma[sigma == 0] = 1e-8

    X_train_scaled = (X_train - mu) / sigma
    X_test_scaled = (X_test - mu) / sigma

    return X_train_scaled, X_test_scaled


def stratified_split(X, y, test_size=0.3):
    # Разделение с сохранением пропорций классов
    classes = np.unique(y)
    train_indices = []
    test_indices = []

    for c in classes:
        c_idx = np.where(y == c)[0]
        np.random.shuffle(c_idx)
        split_point = int(len(c_idx) * (1 - test_size))
        train_indices.extend(c_idx[:split_point])
        test_indices.extend(c_idx[split_point:])

    # Перемешиваем итоговые индексы
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]