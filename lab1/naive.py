"""
Наивная оценка delta_min для квадратной интервальной матрицы 3x3
через интервальный определитель (правило Саррюса + интервальная арифметика IR).

Также: перебор 512 вершинных матриц (критерий Баумана) для сравнения.
"""

import numpy as np
from itertools import product


# =========================================================
# Интервальная арифметика IR (скаляры)
# =========================================================

def iadd(x, y):
    """[x] + [y]"""
    return (x[0] + y[0], x[1] + y[1])

def isub(x, y):
    """[x] - [y]"""
    return (x[0] - y[1], x[1] - y[0])

def imul(x, y):
    """[x] * [y] по таблице Кэли"""
    prods = (x[0]*y[0], x[0]*y[1], x[1]*y[0], x[1]*y[1])
    return (min(prods), max(prods))

def ineg(x):
    """- [x]"""
    return (-x[1], -x[0])


# =========================================================
# Интервальный определитель 3x3 по правилу Саррюса
# =========================================================

def interval_det3(A):
    """
    A — список списков интервалов: A[i][j] = (lo, hi).
    Возвращает интервальный определитель det(A) ∈ IR как (lo, hi).
    Формула Саррюса:
        det = a11 a22 a33 + a12 a23 a31 + a13 a21 a32
            - a13 a22 a31 - a11 a23 a32 - a12 a21 a33
    """
    a11, a12, a13 = A[0]
    a21, a22, a23 = A[1]
    a31, a32, a33 = A[2]

    t1 = imul(imul(a11, a22), a33)
    t2 = imul(imul(a12, a23), a31)
    t3 = imul(imul(a13, a21), a32)
    t4 = imul(imul(a13, a22), a31)
    t5 = imul(imul(a11, a23), a32)
    t6 = imul(imul(a12, a21), a33)

    pos = iadd(iadd(t1, t2), t3)
    neg = iadd(iadd(t4, t5), t6)
    return isub(pos, neg)


def build_interval_matrix(mid, delta):
    """Строит интервальную матрицу [mid ± delta·1]."""
    return [[(mid[i, j] - delta, mid[i, j] + delta)
             for j in range(3)] for i in range(3)]


# =========================================================
# Исходные данные
# =========================================================

midA2 = np.array([
    [1.10, 0.90, 1.10],
    [1.40, 1.00, 0.80],
    [0.80, 1.40, 1.20]
])

print("det(mid A2) =", np.linalg.det(midA2))


# =========================================================
# 1. Наивный delta_min через интервальный определитель
# =========================================================

def naive_delta_min(mid, delta_grid):
    """Первое delta, при котором 0 ∈ det(A(delta))."""
    prev = None
    for delta in delta_grid:
        IA = build_interval_matrix(mid, delta)
        det_lo, det_hi = interval_det3(IA)
        if det_lo <= 0.0 <= det_hi:
            return delta, (det_lo, det_hi), prev
        prev = (delta, (det_lo, det_hi))
    return None, None, None


print("\n" + "=" * 64)
print("1. НАИВНЫЙ delta_min ЧЕРЕЗ ИНТЕРВАЛЬНЫЙ ОПРЕДЕЛИТЕЛЬ")
print("=" * 64)

delta_grid = np.linspace(0.0, 0.2, 20001)   # шаг 1e-5
d_naive, det_at_min, prev = naive_delta_min(midA2, delta_grid)

if prev is not None:
    d_prev, det_prev = prev
    print(f"  На шаге delta = {d_prev:.5f}:  det = [{det_prev[0]:.5f}, {det_prev[1]:.5f}]"
          f"  (нуль ещё не достигнут)")
print(f"  Наивная delta_min ≈ {d_naive:.5f}")
print(f"  При этом  det(A(delta_min)) = [{det_at_min[0]:.5f}, {det_at_min[1]:.5f}]"
      f"  — нуль внутри")
print(f"  ИСТИННОЕ delta_min (из бисекции поверх BFGS) = 0.10000")
print(f"  Занижение в {0.1/d_naive:.2f} раз")


# =========================================================
# 2. Контроль: перебор вершинных матриц (критерий Баумана)
#    Определители вершинных матриц A' ∈ vert A при разных delta.
#    Первое delta, при котором появляется det = 0  →  «истинное» по вершинам.
# =========================================================

print("\n" + "=" * 64)
print("2. КОНТРОЛЬ: КРИТЕРИЙ БАУМАНА (перебор 512 вершин)")
print("=" * 64)

def vertex_det_signs(mid, delta):
    """Считает знаки определителей всех 512 вершинных матриц A(delta)."""
    n = 3
    signs = []
    for signs_bits in product([-1, 1], repeat=n * n):
        A = np.empty((n, n))
        k = 0
        for i in range(n):
            for j in range(n):
                A[i, j] = mid[i, j] + signs_bits[k] * delta
                k += 1
        signs.append(np.sign(np.linalg.det(A)))
    return signs


d_min_baumann = None
for delta in delta_grid:
    s = vertex_det_signs(midA2, delta)
    # Особенность: определители имеют разные знаки, либо есть ровно нулевые
    if any(x == 0 for x in s) or (min(s) < 0 < max(s)):
        d_min_baumann = delta
        break

print(f"  Критерий Баумана: delta_min ≈ {d_min_baumann:.5f}")
print(f"  (совпадает с бисекцией поверх BFGS, если = 0.1)")


# =========================================================
# 3. Диагностика: как меняется det(A(delta)) с ростом delta
# =========================================================

print("\n" + "=" * 64)
print("3. ДИАГНОСТИКА: интервальный определитель при разных delta")
print("=" * 64)
print(f"  {'delta':>8} | {'det_lo':>10} | {'det_hi':>10} | содержит 0?")
print("  " + "-" * 50)

for delta in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]:
    IA = build_interval_matrix(midA2, delta)
    lo, hi = interval_det3(IA)
    contains = "ДА" if lo <= 0 <= hi else "нет"
    print(f"  {delta:>8.4f} | {lo:>10.5f} | {hi:>10.5f} | {contains:>9}")