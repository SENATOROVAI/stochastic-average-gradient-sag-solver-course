import numpy as np


def sag(
    X,
    y,
    eta=0.01,
    n_iters=1000,
    w_init=None,
    random_state=42,
):
    """
    Чистая реализация SAG для MSE.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    eta : float
        Шаг обучения (должен быть <= 1 / L)
    n_iters : int
        Число итераций
    w_init : ndarray or None
        Начальное значение весов
    random_state : int

    Returns
    -------
    w : ndarray
        Обученные веса
    history : list
        Значения ||w|| по итерациям (для диагностики)
    """

    rng = np.random.default_rng(random_state)

    n, d = X.shape

    # Инициализация весов
    if w_init is None:
        w = np.zeros(d)
    else:
        w = w_init.copy()

    # Память градиентов: по одному на объект
    grad_memory = np.zeros((n, d))

    # Усреднённый градиент (сумма последних градиентов)
    d_avg = np.zeros(d)

    history = []

    for k in range(n_iters):
        # 1. случайно выбираем объект
        i = rng.integers(0, n)

        x_i = X[i]
        y_i = y[i]

        # 2. считаем новый градиент
        residual = x_i @ w - y_i
        g_new = residual * x_i

        # 3. обновляем средний градиент
        d_avg -= grad_memory[i]
        d_avg += g_new

        # 4. обновляем память
        grad_memory[i] = g_new

        # 5. шаг спуска
        w -= eta * d_avg / n

        history.append(np.linalg.norm(w))

    return w, history
