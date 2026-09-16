"""
Лабораторная работа №1 по интервальному анализу.
Поиск delta_min: A1 (3x2) — неполный столбцовый ранг; A2 (3x3) — det = 0.
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


# ==================================================================
# Общие утилиты
# ==================================================================

def check_membership(A_prime, midA, radA, tol=1e-9):
    """Проверка: принадлежит ли A' интервальной матрице A = [midA ± radA]."""
    return bool(np.all(np.abs(A_prime - midA) <= radA + tol))


# ==================================================================
# ЧАСТЬ 1. Прямоугольная матрица A1 ∈ R^{3×2}
# ==================================================================

midA1 = np.array([
    [0.95, 1.00],
    [1.05, 1.00],
    [1.10, 1.00]
])

# Режим томографии:  rad A1 = delta * ones(3,2)
# Режим регрессии:   rad A1 = delta * [[1,0],[1,0],[1,0]]


def solve_A1(mode, c_grid):
    """
    Для каждого c ∈ c_grid находим минимальное delta, при котором
    столбцы могут стать линейно зависимыми: A'_i2 = c * A'_i1.

      томография: delta_i(c) = |c*a_i - b_i| / (1 + |c|)
      регрессия:  A'_i2 = b_i = const, A'_i1 = b_i / c,
                  delta_i(c) = |b_i / c - a_i|

    delta_min(c) = max_i delta_i(c), затем минимум по c.
    """
    best_delta, best_c = np.inf, None
    for c in c_grid:
        if abs(c) < 1e-9:
            continue
        if mode == "tomography":
            deltas = np.abs(c * midA1[:, 0] - midA1[:, 1]) / (1.0 + abs(c))
        else:
            deltas = np.abs(midA1[:, 1] / c - midA1[:, 0])
        d = float(np.max(deltas))
        if d < best_delta:
            best_delta, best_c = d, float(c)

    delta_min, c_opt = best_delta, best_c

    # Строим точечную матрицу неполного ранга
    A1p = np.zeros((3, 2))
    if mode == "tomography":
        for i in range(3):
            a, b = midA1[i]
            lo1, hi1 = a - delta_min, a + delta_min
            if c_opt > 0:
                lo2, hi2 = (b - delta_min) / c_opt, (b + delta_min) / c_opt
            else:
                lo2, hi2 = (b + delta_min) / c_opt, (b - delta_min) / c_opt
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            x = 0.5 * (lo + hi)
            A1p[i] = [x, c_opt * x]
    else:
        A1p[:, 0] = midA1[:, 1] / c_opt
        A1p[:, 1] = midA1[:, 1]
    return delta_min, c_opt, A1p


print("=" * 64)
print("ЧАСТЬ 1. Прямоугольная матрица A1 (3x2)")
print("=" * 64)

c_grid = np.linspace(0.5, 1.5, 200_001)

for mode, name in [("tomography", "ТОМОГРАФИЯ"),
                   ("regression", "РЕГРЕССИЯ")]:
    delta, c, A1p = solve_A1(mode, c_grid)
    if mode == "tomography":
        radA1 = delta * np.ones_like(midA1)
    else:
        radA1 = delta * np.array([[1., 0.], [1., 0.], [1., 0.]])
    ok = check_membership(A1p, midA1, radA1)
    r = np.linalg.matrix_rank(A1p)

    print(f"\n--- Режим: {name} ---")
    print(f"  delta_min   = {delta:.6f}")
    print(f"  c_opt       = {c:.5f}")
    print(f"  A1' =\n{A1p}")
    print(f"  rank(A1')   = {r}")
    print(f"  A1' ∈ A1    : {ok}")


# ==================================================================
# ЧАСТЬ 2. Квадратная матрица A2 ∈ R^{3×3}
# ==================================================================

midA2 = np.array([
    [1.10, 0.90, 1.10],
    [1.40, 1.00, 0.80],
    [0.80, 1.40, 1.20]
])

print("\n" + "=" * 64)
print("ЧАСТЬ 2. Квадратная матрица A2 (3x3)")
print("=" * 64)
print(f"\ndet(mid A2) = {np.linalg.det(midA2):.6f}")


def min_abs_det(delta, n_starts=40, seed=1):
    """Минимизируем |det(M + delta * S)| по S ∈ [-1, 1]^{3×3} (L-BFGS-B)."""
    rng = np.random.default_rng(seed)

    def f(s):
        S = s.reshape(3, 3)
        return abs(np.linalg.det(midA2 + delta * S))

    bounds = [(-1.0, 1.0)] * 9
    best_f, best_x = np.inf, None
    for _ in range(n_starts):
        x0 = rng.uniform(-1.0, 1.0, 9)
        res = minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-16, "gtol": 1e-14,
                                "maxiter": 5000})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x
    return best_f, best_x


# -------- Бисекция по delta ---------------------------------------
lo, hi = 0.0, 1.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    val, _ = min_abs_det(mid)
    if val < 1e-9:
        hi = mid
    else:
        lo = mid

delta_min_A2 = hi
print(f"\ndelta_min(A2) = {delta_min_A2:.6f}")

# -------- Строим вырожденную матрицу ------------------------------
val, s_opt = min_abs_det(delta_min_A2, n_starts=200)
S_opt = s_opt.reshape(3, 3)
A2p = midA2 + delta_min_A2 * S_opt
radA2 = delta_min_A2 * np.ones((3, 3))
ok = check_membership(A2p, midA2, radA2)

print(f"\n  A2' =\n{A2p}")
print(f"  det(A2')    = {np.linalg.det(A2p):.3e}")
print(f"  A2' ∈ A2    : {ok}")


# ==================================================================
# Сводная таблица результатов
# ==================================================================

print("\n" + "=" * 64)
print("СВОДНАЯ ТАБЛИЦА")
print("=" * 64)

d_tomo, c_tomo, _ = solve_A1("tomography", c_grid)
d_reg, c_reg, _ = solve_A1("regression", c_grid)

print(f"  A1, томография : delta_min = {d_tomo:.6f}  (c_opt = {c_tomo:.5f}, rank < 2)")
print(f"  A1, регрессия  : delta_min = {d_reg:.6f}   (c_opt = {c_reg:.5f}, rank < 2)")
print(f"  A2             : delta_min = {delta_min_A2:.6f}  (det = 0)")