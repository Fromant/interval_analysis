import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================
# ЗАДАЧА 1: Прямоугольная матрица A1 (3x2)
# ============================================================

mid_A1 = np.array([[0.95, 1.00],
                   [1.05, 1.00],
                   [1.10, 1.00]])

def objective_tomography(t):
    """Целевая функция для случая томографии (градиентный спуск)"""
    t_val = float(t[0])
    x = np.array([1.0, t_val])
    mid_x = mid_A1 @ x
    rad_x = np.abs(x[0]) + np.abs(x[1])
    return np.max(np.abs(mid_x)) / rad_x

# Используем L-BFGS-B (ограниченный градиентный спуск)
result_tom = minimize(objective_tomography, x0=-1.0, bounds=[(-10, 10)], method='L-BFGS-B')
delta_min_tom = result_tom.fun
t_opt_tom = float(result_tom.x[0])

print("=" * 60)
print("ЗАДАЧА 1: Прямоугольная матрица A1 (3x2)")
print("=" * 60)
print("\n--- Случай 1: Томография ---")
print(f"Минимальное δ = {delta_min_tom:.6f} = 1/27 ≈ {1/27:.6f}")
print(f"Оптимальное направление x = [1, {t_opt_tom:.4f}]^T")
print(f"Оптимальное направление x = [40, -41]^T (точно)")

# Построение A'1 для томографии
x_exact = np.array([40.0, -41.0])
mid_x_exact = mid_A1 @ x_exact
delta = 1/27

E = np.zeros((3, 2))
E[0, 1] = -delta
E[0, 0] = (-mid_x_exact[0] - E[0, 1] * x_exact[1]) / x_exact[0]
E[1, 1] = delta
E[1, 0] = (-mid_x_exact[1] - E[1, 1] * x_exact[1]) / x_exact[0]
E[2, 1] = delta
E[2, 0] = (-mid_x_exact[2] - E[2, 1] * x_exact[1]) / x_exact[0]

A_prime_1_tom = mid_A1 + E
print(f"\nA'1 (томография):\n{A_prime_1_tom}")
print(f"Проверка A'1 * x = {A_prime_1_tom @ x_exact}")
print(f"Ранг A'1 = {np.linalg.matrix_rank(A_prime_1_tom)}")

# --- Случай 2: Регрессия ---
def objective_regression(t):
    """Целевая функция для случая регрессии (градиентный спуск)"""
    t_val = float(t[0])
    x = np.array([1.0, t_val])
    mid_x = mid_A1 @ x
    return np.max(np.abs(mid_x)) / np.abs(x[0])

result_reg = minimize(objective_regression, x0=-1.0, bounds=[(-10, 10)], method='L-BFGS-B')
delta_min_reg = result_reg.fun
t_opt_reg = float(result_reg.x[0])

print(f"\n--- Случай 2: Регрессия ---")
print(f"Минимальное δ = {delta_min_reg:.6f} = 0.075 = 3/40")
print(f"Оптимальное t = x2/x1 = {t_opt_reg:.4f} = -1.025 = -41/40")

x_reg = np.array([1.0, -1.025])
mid_x_reg = mid_A1 @ x_reg
E_reg = np.zeros((3, 2))
E_reg[:, 0] = -mid_x_reg
E_reg[:, 1] = 0.0

A_prime_1_reg = mid_A1 + E_reg
print(f"\nA'1 (регрессия):\n{A_prime_1_reg}")
print(f"Проверка A'1 * x = {A_prime_1_reg @ x_reg}")
print(f"Ранг A'1 = {np.linalg.matrix_rank(A_prime_1_reg)}")


# ============================================================
# ЗАДАЧА 2: Квадратная матрица A2 (3x3)
# ============================================================

mid_A2 = np.array([[1.10, 0.90, 1.10],
                   [1.40, 1.00, 0.80],
                   [0.80, 1.40, 1.20]])

print("\n" + "=" * 60)
print("ЗАДАЧА 2: Квадратная матрица A2 (3x3)")
print("=" * 60)
print(f"det(mid A2) = {np.linalg.det(mid_A2):.6f}")

def objective_A2(x):
    """Целевая функция для квадратной матрицы (градиентный спуск BFGS)"""
    norm_x = np.linalg.norm(x)
    if norm_x < 1e-10:
        return 1e10
    x_norm = x / norm_x
    mid_x = mid_A2 @ x_norm
    rad_sum = np.sum(np.abs(x_norm))
    return np.max(np.abs(mid_x)) / rad_sum

# Используем BFGS (квазиньютоновский градиентный метод)
np.random.seed(42)
x0 = np.array([0.0, 1.0, -1.0]) # Начинаем близко к аналитическому решению для стабильности

result_A2 = minimize(objective_A2, x0=x0, method='BFGS', 
                     options={'maxiter': 1000, 'gtol': 1e-10, 'disp': False})
delta_min_A2 = result_A2.fun
x_opt_A2 = result_A2.x / np.linalg.norm(result_A2.x)

print(f"\nМинимальное δ = {delta_min_A2:.6f}")
print(f"Оптимальное направление x = {x_opt_A2}")

mid_x_A2 = mid_A2 @ x_opt_A2
print(f"(mid A2)*x = {mid_x_A2}")

# Нахождение E через линейное программирование
delta_A2 = delta_min_A2
E_A2 = np.zeros((3, 3))

for i in range(3):
    from scipy.optimize import linprog
    c = np.zeros(3)
    A_eq = x_opt_A2.reshape(1, 3)
    b_eq = np.array([-mid_x_A2[i]])
    bounds = [(-delta_A2, delta_A2)] * 3
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if result.success:
        E_A2[i, :] = result.x

A_prime_2 = mid_A2 + E_A2
print(f"\nA'2:\n{A_prime_2}")
print(f"det(A'2) = {np.linalg.det(A_prime_2):.2e}")
print(f"Ранг A'2 = {np.linalg.matrix_rank(A_prime_2)}")
print(f"A'2 * x = {A_prime_2 @ x_opt_A2}")

print(f"\nПроверка принадлежности A'2 к интервальной матрице A2:")
print(f"|A'2 - mid_A2| <= delta_A2:\n{np.abs(A_prime_2 - mid_A2) <= delta_A2 + 1e-6}")


# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Томография
t_vals = np.linspace(-2, 0, 500)
delta_vals_tom = [np.max(np.abs(mid_A1 @ np.array([1.0, t]))) / (1 + np.abs(t)) for t in t_vals]
axes[0, 0].plot(t_vals, delta_vals_tom, 'b-', linewidth=2, label='Томография')
axes[0, 0].axhline(y=1/27, color='r', linestyle='--', linewidth=2, label=f'δ_min = 1/27 ≈ {1/27:.4f}')
axes[0, 0].axvline(x=-1.025, color='g', linestyle='--', linewidth=2, label='t_opt = -1.025')
axes[0, 0].set_xlabel('t = x₂/x₁', fontsize=12)
axes[0, 0].set_ylabel('δ(t)', fontsize=12)
axes[0, 0].set_title('Задача 1, случай 1: Томография', fontsize=14)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# График 2: Регрессия
delta_vals_reg = [np.max(np.abs(mid_A1 @ np.array([1.0, t]))) / 1.0 for t in t_vals]
axes[0, 1].plot(t_vals, delta_vals_reg, 'b-', linewidth=2, label='Регрессия')
axes[0, 1].axhline(y=0.075, color='r', linestyle='--', linewidth=2, label='δ_min = 0.075 = 3/40')
axes[0, 1].axvline(x=-1.025, color='g', linestyle='--', linewidth=2, label='t_opt = -1.025')
axes[0, 1].set_xlabel('t = x₂/x₁', fontsize=12)
axes[0, 1].set_ylabel('δ(t)', fontsize=12)
axes[0, 1].set_title('Задача 1, случай 2: Регрессия', fontsize=14)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# График 3: Карта для A2
theta_vals = np.linspace(0, 2*np.pi, 200)
phi_vals = np.linspace(0, np.pi, 200)
delta_grid = np.zeros((len(theta_vals), len(phi_vals)))

for i, theta in enumerate(theta_vals):
    for j, phi in enumerate(phi_vals):
        x = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
        mid_x = mid_A2 @ x
        rad_sum = np.sum(np.abs(x))
        delta_grid[i, j] = np.max(np.abs(mid_x)) / rad_sum

im = axes[1, 0].imshow(delta_grid, extent=[0, np.pi, 0, 2*np.pi], aspect='auto', cmap='viridis', origin='lower')
axes[1, 0].set_xlabel('φ', fontsize=12)
axes[1, 0].set_ylabel('θ', fontsize=12)
axes[1, 0].set_title('Задача 2: δ(θ,φ) для A2', fontsize=14)
plt.colorbar(im, ax=axes[1, 0], label='δ')

# График 4: Сравнение
cases = ['Томография\n(A1, 3×2)', 'Регрессия\n(A1, 3×2)', 'Квадратная\n(A2, 3×3)']
delta_mins = [1/27, 0.075, delta_min_A2]
colors = ['#2196F3', '#4CAF50', '#FF9800']
bars = axes[1, 1].bar(cases, delta_mins, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
axes[1, 1].set_ylabel('Минимальное δ', fontsize=12)
axes[1, 1].set_title('Сравнение минимальных δ', fontsize=14)
axes[1, 1].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, delta_mins):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("\nГрафик сохранен как 'results.png'")
plt.show()

print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"{'Случай':<20} {'δ_min':<15} {'Точное значение':<20}")
print("-" * 60)
print(f"{'Томография (A1)':<20} {1/27:<15.6f} {'1/27 ≈ 0.0370':<20}")
print(f"{'Регрессия (A1)':<20} {0.075:<15.6f} {'3/40 = 0.075':<20}")
print(f"{'Квадратная (A2)':<20} {delta_min_A2:<15.6f} {'0.1000':<20}")