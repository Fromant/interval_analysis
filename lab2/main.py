import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================================
# КЛАСС ИНТЕРВАЛЬНОЙ АРИФМЕТИКИ
# ============================================================================

class Interval:
    def __init__(self, a, b):
        self.a = float(min(a, b))
        self.b = float(max(a, b))

    def __repr__(self):
        return f"[{self.a:.6f}, {self.b:.6f}]"

    def mid(self):
        return (self.a + self.b) / 2.0

    def rad(self):
        return (self.b - self.a) / 2.0

    def wid(self):
        return self.b - self.a

    # ---------- сложение ----------
    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.a + other.a, self.b + other.b)
        return Interval(self.a + other, self.b + other)

    def __radd__(self, other):
        return self.__add__(other)

    # ---------- вычитание ----------
    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.a - other.b, self.b - other.a)
        return Interval(self.a - other, self.b - other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Interval(other - self.b, other - self.a)
        raise TypeError

    # ---------- умножение ----------
    def __mul__(self, other):
        if isinstance(other, Interval):
            prods = [self.a * other.a, self.a * other.b,
                     self.b * other.a, self.b * other.b]
            return Interval(min(prods), max(prods))
        if other >= 0:
            return Interval(self.a * other, self.b * other)
        return Interval(self.b * other, self.a * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    # ---------- деление ----------
    def __truediv__(self, other):
        if isinstance(other, Interval):
            if other.a <= 0 <= other.b:
                raise ValueError("Деление на интервал, содержащий 0")
            return self * Interval(1.0 / other.b, 1.0 / other.a)
        if other >= 0:
            return Interval(self.a / other, self.b / other)
        return Interval(self.b / other, self.a / other)

    def __neg__(self):
        return Interval(-self.b, -self.a)

    # ---------- функции ----------
    def exp(self):
        return Interval(np.exp(self.a), np.exp(self.b))

    def __pow__(self, n):
        if n == 2:
            if self.a >= 0:
                return Interval(self.a ** 2, self.b ** 2)
            elif self.b <= 0:
                return Interval(self.b ** 2, self.a ** 2)
            else:
                return Interval(0.0, max(self.a ** 2, self.b ** 2))
        raise NotImplementedError

    # ---------- пересечение и оболочка ----------
    def intersect(self, other):
        return Interval(max(self.a, other.a), min(self.b, other.b))

    def hull(self, other):
        return Interval(min(self.a, other.a), max(self.b, other.b))


def hausdorff(F, ran):
    return max(abs(F.a - ran[0]), abs(F.b - ran[1]))


# ============================================================================
#  f2(x) = x * exp(-x^2)
# ============================================================================

def f2(x):
    return x * np.exp(-x ** 2)

def f2_prime(x):
    return np.exp(-x ** 2) * (1.0 - 2.0 * x ** 2)

X_f2 = [-1.5, 1.5]
X = Interval(*X_f2)

print("=" * 70)
print("f2(x) = x * exp(-x^2),   X =", X)
print("=" * 70)

# A. Точная область значений
xf = np.linspace(X.a, X.b, 200_000)
yf = f2(xf)
ran_f2 = [float(np.min(yf)), float(np.max(yf))]
print(f"\nA. ran(f2, X) = [{ran_f2[0]:.6f}, {ran_f2[1]:.6f}]")

# C.1 Константа Липшица
L_f2 = float(np.max(np.abs(f2_prime(xf))))
print(f"C.1 L = {L_f2:.6f}")

# B.1 Естественное расширение  x * exp(-(x^2))
#     x^2 через __pow__ → [0, 2.25]
Xsq = X ** 2                       # [0, 2.25]
f2_B1 = X * (-Xsq).exp()
print(f"\nB.1 Естественное расширение:       {f2_B1}")

# B.2 Эквивалентное выражение  x * exp(-(x*x))
#     x*x через интервальное умножение → [-2.25, 2.25]
Xsq_naive = X * X                  # [-2.25, 2.25]
f2_B2 = X * (-Xsq_naive).exp()
print(f"B.2 Через X*X (эффект зависимости): {f2_B2}")

# B.3 Дифференциальная центрированная форма
def f2_mv(X, c):
    fc = f2(c)
    xv = np.linspace(X.a, X.b, 50_000)
    fp = f2_prime(xv)
    fpX = Interval(float(np.min(fp)), float(np.max(fp)))
    return fc + fpX * (X - c)

print("\nB.3 Дифференциальная центрированная форма:")
for c in [X.mid(), 0.5, -0.5, X.a, X.b]:
    r = f2_mv(X, c)
    print(f"  c = {c:+.4f}:  {r}   dist = {hausdorff(r, ran_f2):.6f}")

# B.4 Наклонная центрированная форма
def f2_sl(X, c):
    fc = f2(c)
    xv = np.linspace(X.a, X.b, 50_000)
    mask = np.abs(xv - c) > 1e-12
    xv = xv[mask]
    slopes = (f2(xv) - fc) / (xv - c)
    sX = Interval(float(np.min(slopes)), float(np.max(slopes)))
    return fc + sX * (X - c)

print("\nB.4 Наклонная центрированная форма:")
for c in [X.mid(), 0.5, -0.5]:
    r = f2_sl(X, c)
    print(f"  c = {c:+.4f}:  {r}   dist = {hausdorff(r, ran_f2):.6f}")

# B.5 Бицентрированная среднезначная форма (теорема Бауманна)
def f2_bic(X):
    xv = np.linspace(X.a, X.b, 50_000)
    fp = f2_prime(xv)
    fpX = Interval(float(np.min(fp)), float(np.max(fp)))
    m, r = fpX.mid(), fpX.rad()
    p = np.clip(m / r, -1, 1) if r > 0 else 0.0
    cs  = X.mid() - p * X.rad()
    csb = X.mid() + p * X.rad()
    return f2_mv(X, cs).intersect(f2_mv(X, csb)), cs, csb

f2_B5, cs2, csb2 = f2_bic(X)
print(f"\nB.5 Бицентрированная форма: {f2_B5}")
print(f"    c* = {cs2:.6f},  c̄* = {csb2:.6f}")

# C. Сводка расстояний
print("\nC. Расстояния по Хаусдорфу:")
for name, F in [
    ("B.1 естеств.", f2_B1), ("B.2 X*X", f2_B2),
    ("B.3 c=mid", f2_mv(X, X.mid())), ("B.4 c=mid", f2_sl(X, X.mid())),
    ("B.5 бицентр.", f2_B5)
]:
    print(f"  {name:20s}  {F}   dist = {hausdorff(F, ran_f2):.6f}")

print(f"\nC.2 Теоретическая оценка: rad F ≤ L·rad X = {L_f2:.4f}·{X.rad():.4f} = {L_f2 * X.rad():.4f}")
print(f"    Фактическая rad B.1: {f2_B1.rad():.4f}")

# ============================================================================
#  f3(x, y) = 2x² + y² − xy − x
# ============================================================================

def f3(x, y):   return 2*x**2 + y**2 - x*y - x
def f3_dx(x, y): return 4*x - y - 1
def f3_dy(x, y): return 2*y - x

X1 = Interval(-1, 2)
X2 = Interval(-2, 2)

print("\n" + "=" * 70)
print(f"f3(x,y) = 2x² + y² − xy − x,   X = {X1} × {X2}")
print("=" * 70)

# E.2
xv = np.linspace(X1.a, X1.b, 500)
yv = np.linspace(X2.a, X2.b, 500)
Xg, Yg = np.meshgrid(xv, yv)
Zg = f3(Xg, Yg)
ran_f3 = [float(np.min(Zg)), float(np.max(Zg))]
print(f"\nE.2 ran(f3, X) = [{ran_f3[0]:.6f}, {ran_f3[1]:.6f}]")

# E.3
def f3_nat(A, B):
    return 2 * (A ** 2) + (B ** 2) - (A * B) - A

f3_E3 = f3_nat(X1, X2)
print(f"E.3 Естественное расширение: {f3_E3}")

# E.4  x(2x − y − 1) + y²
def f3_eqv(A, B):
    return A * (2 * A - B - 1) + B ** 2

f3_E4 = f3_eqv(X1, X2)
print(f"E.4 Эквивалентное выражение: {f3_E4}")

# E.5
def f3_mv(A, B, c1, c2):
    fc = f3(c1, c2)
    xv = np.linspace(A.a, A.b, 300)
    yv = np.linspace(B.a, B.b, 300)
    Xg, Yg = np.meshgrid(xv, yv)
    dxX = Interval(float(np.min(f3_dx(Xg, Yg))), float(np.max(f3_dx(Xg, Yg))))
    dyX = Interval(float(np.min(f3_dy(Xg, Yg))), float(np.max(f3_dy(Xg, Yg))))
    return fc + dxX * (A - c1) + dyX * (B - c2)

print("\nE.5 Дифференциальная центрированная форма:")
for c1, c2 in [(X1.mid(), X2.mid()), (0.0, 0.0), (X1.a, X2.a)]:
    r = f3_mv(X1, X2, c1, c2)
    print(f"  c = ({c1:+.4f}, {c2:+.4f}):  {r}   dist = {hausdorff(r, ran_f3):.6f}")

# E.6
def f3_bic(A, B):
    xv = np.linspace(A.a, A.b, 300)
    yv = np.linspace(B.a, B.b, 300)
    Xg, Yg = np.meshgrid(xv, yv)
    dxX = Interval(float(np.min(f3_dx(Xg, Yg))), float(np.max(f3_dx(Xg, Yg))))
    dyX = Interval(float(np.min(f3_dy(Xg, Yg))), float(np.max(f3_dy(Xg, Yg))))
    def baum(df, I):
        m, r = df.mid(), df.rad()
        p = np.clip(m / r, -1, 1) if r > 0 else 0.0
        return I.mid() - p * I.rad(), I.mid() + p * I.rad()
    c1s, c1sb = baum(dxX, A)
    c2s, c2sb = baum(dyX, B)
    mv1 = f3_mv(A, B, c1s, c2s)
    mv2 = f3_mv(A, B, c1sb, c2sb)
    return mv1.intersect(mv2), (c1s, c2s), (c1sb, c2sb)

f3_E6, cs3, csb3 = f3_bic(X1, X2)
print(f"\nE.6 Бицентрированная форма: {f3_E6}")
print(f"    c* = ({cs3[0]:.4f}, {cs3[1]:.4f}),  c̄* = ({csb3[0]:.4f}, {csb3[1]:.4f})")

# E.7
print("\nE.7 Дробление бруса:")
X1L, X1R = Interval(X1.a, X1.mid()), Interval(X1.mid(), X1.b)
f3_L, f3_R = f3_nat(X1L, X2), f3_nat(X1R, X2)
hull_x = f3_L.hull(f3_R)
print(f"  По x:  {f3_L}  ∪  {f3_R}  →  {hull_x}   dist = {hausdorff(hull_x, ran_f3):.6f}")

X2L, X2R = Interval(X2.a, X2.mid()), Interval(X2.mid(), X2.b)
f3_B, f3_T = f3_nat(X1, X2L), f3_nat(X1, X2R)
hull_y = f3_B.hull(f3_T)
print(f"  По y:  {f3_B}  ∪  {f3_T}  →  {hull_y}   dist = {hausdorff(hull_y, ran_f3):.6f}")

# E.8
dx_all = f3_dx(Xg, Yg)
dy_all = f3_dy(Xg, Yg)
L1 = float(np.max(np.abs(dx_all)))
L2 = float(np.max(np.abs(dy_all)))
print(f"\nE.8 L1 = {L1:.4f},  L2 = {L2:.4f}")
theo = L1 * X1.rad() + L2 * X2.rad()
print(f"    Теоретическая оценка: {L1:.1f}·{X1.rad():.1f} + {L2:.1f}·{X2.rad():.1f} = {theo:.4f}")
print(f"    Фактическая rad E.3: {f3_E3.rad():.4f}")

# E.9 Сводка
print("\nE.9 Расстояния по Хаусдорфу:")
for name, F in [
    ("E.3 естеств.", f3_E3), ("E.4 эквив.", f3_E4),
    ("E.5 c=mid", f3_mv(X1, X2, X1.mid(), X2.mid())),
    ("E.6 бицентр.", f3_E6),
    ("E.7 дробл. x", hull_x), ("E.7 дробл. y", hull_y)
]:
    print(f"  {name:20s}  {F}   dist = {hausdorff(F, ran_f3):.6f}")

# ============================================================================
#  Графики
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(xf, yf, 'b-', lw=2)
axes[0].axhline(ran_f2[0], color='r', ls='--', label=f'ran = [{ran_f2[0]:.4f}, {ran_f2[1]:.4f}]')
axes[0].axhline(ran_f2[1], color='r', ls='--')
axes[0].set(xlabel='x', ylabel='f₂(x)', title='f₂(x) = x·exp(−x²)')
axes[0].legend(); axes[0].grid(True)

x3 = np.linspace(-1, 2, 100); y3 = np.linspace(-2, 2, 100)
X3, Y3 = np.meshgrid(x3, y3); Z3 = f3(X3, Y3)
cf = axes[1].contourf(X3, Y3, Z3, 20, cmap=cm.coolwarm)
axes[1].contour(X3, Y3, Z3, 20, colors='k', lw=0.5)
axes[1].set(xlabel='x', ylabel='y', title='f₃(x,y) = 2x² + y² − xy − x')
fig.colorbar(cf, ax=axes[1])
plt.tight_layout(); plt.savefig('f2_f3_plots.png', dpi=200)

fig2 = plt.figure(figsize=(10, 7))
ax = fig2.add_subplot(111, projection='3d')
ax.plot_surface(X3, Y3, Z3, cmap=cm.coolwarm, alpha=0.8)
ax.set(xlabel='x', ylabel='y', zlabel='z', title='f₃(x,y)')
plt.savefig('f3_surface.png', dpi=200)
print("\nГрафики сохранены.")