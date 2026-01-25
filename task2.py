import numpy as np


def saga(
    X,
    y,
    eta=0.01,
    n_iters=1000,
    w_init=None,
    random_state=42,
):
    """
    Чистая реализация SAGA для MSE (линейная регрессия).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    eta : float
        Шаг обучения (обычно <= 1 / (3L))
    n_iters : int
        Число итераций
    w_init : ndarray or None
        Начальные веса
    random_state : int

    Returns
    -------
    w : ndarray
        Обученные веса
    history : list
        ||w|| по итерациям
    """

    rng = np.random.default_rng(random_state)

    n, d = X.shape

    # Инициализация весов
    if w_init is None:
        w = np.zeros(d)
    else:
        w = w_init.copy()

    # Память градиентов
    grad_memory = np.zeros((n, d))

    # Средний градиент
    avg_grad = np.zeros(d)

    history = []

    for k in range(n_iters):
        # 1. случайный индекс
        i = rng.integers(0, n)

        x_i = X[i]
        y_i = y[i]

        # 2. новый градиент
        residual = x_i @ w - y_i
        g_new = residual * x_i

        # 3. SAGA-обновление (несмещённое!)
        update = g_new - grad_memory[i] + avg_grad

        # 4. шаг спуска
        w -= eta * update

        # 5. обновляем средний градиент
        avg_grad += (g_new - grad_memory[i]) / n

        # 6. обновляем память
        grad_memory[i] = g_new

        history.append(np.linalg.norm(w))

    return w, history
