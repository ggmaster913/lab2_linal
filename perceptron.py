import numpy as np


class Perceptron:
    def __init__(self, input_dim, init_type='small_random', l2_coef=0.0):
        # l2_coef - это лямбда для регуляризации, чтобы веса не улетали в космос
        self.l2_coef = l2_coef

        # Инициализация весов как просили в задании
        if init_type == 'zero':
            self.w = np.zeros(input_dim)
        elif init_type == 'large':
            self.w = np.random.normal(0, 10, input_dim)
        else:
            # small_random по умолчанию (небольшие случайные числа)
            self.w = np.random.randn(input_dim) * 0.01

        self.b = 0.0

        # Переменные для Momentum (чтобы спуск не сильно "шатало")
        self.v_w = np.zeros(input_dim)
        self.v_b = 0.0

    def sigmoid(self, z):
        # Ограничиваем z, иначе np.exp может выдать overflow
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))  # формула из теории

    def forward(self, X):
        # z = w^T * x + b
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred, loss_type='bce'):
        # Чтобы не было логарифма от нуля
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # Штраф за большие веса (L2)
        l2_penalty = (self.l2_coef / 2) * np.sum(self.w ** 2)

        if loss_type == 'bce':
            # Бинарная кросс-энтропия (log-loss)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            return loss + l2_penalty
        elif loss_type == 'hinge':
            # Бонусное задание: Hinge loss (метки должны быть -1 и 1)
            y_hinge = np.where(y_true == 0, -1, 1)
            z = np.dot(X, self.w) + self.b  # тут нужен сырой z, без сигмоиды
            loss = np.mean(np.maximum(0, 1 - y_hinge * z))
            return loss + l2_penalty

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, lr=0.1, batch_size=32, momentum_beta=0.0):

        train_losses = []
        val_losses = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Перемешиваем данные перед каждой эпохой, чтобы сеть не зубрила порядок
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0

            # Разбиваем на мини-батчи
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                m = X_batch.shape[0]

                # Прямой проход
                y_pred = self.forward(X_batch)

                # Градиенты по формулам из методички: ошибка * вход
                error = y_pred - y_batch
                dw = (1 / m) * np.dot(X_batch.T, error) + self.l2_coef * self.w
                db = (1 / m) * np.sum(error)

                # Обновление весов (с моментумом, если beta > 0)
                self.v_w = momentum_beta * self.v_w + lr * dw
                self.v_b = momentum_beta * self.v_b + lr * db

                self.w -= self.v_w
                self.b -= self.v_b

            # Считаем loss для графиков
            train_loss = self.compute_loss(y_train, self.forward(X_train))
            train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(y_val, self.forward(X_val))
                val_losses.append(val_loss)

        return train_losses, val_losses

    def predict(self, X, threshold=0.5):
        # Если вероятность > 0.5, то класс 1, иначе 0
        return (self.forward(X) >= threshold).astype(int)