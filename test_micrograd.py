# thank u for the test file claude!!

import math
import random
import torch
from micrograd_finished import Value, MLP

TOL = 1e-5
passed = failed = 0


def check(name, mine, theirs, tol=TOL):
    global passed, failed
    ok = abs(mine - theirs) < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} mine={mine:+.8f}  torch={theirs:+.8f}")
    if ok:
        passed += 1
    else:
        failed += 1


def compare_to_torch(name, build_mine, build_torch, inits):
    """
    build_mine(*values) -> output Value
    build_torch(*tensors) -> output tensor
    inits: list of floats, the leaf values
    Compares forward output and every leaf gradient.
    """
    print(f"\n{name}")
    vs = [Value(v) for v in inits]
    out = build_mine(*vs)
    out.backward()

    ts = [torch.tensor(float(v), requires_grad=True, dtype=torch.double) for v in inits]
    tout = build_torch(*ts)
    tout.backward()

    check("forward", out.data, tout.item())
    for i, (v, t) in enumerate(zip(vs, ts)):
        check(f"d/d(leaf {i})", v.grad, t.grad.item())


# ---------------------------------------------------------------- basic ops
compare_to_torch(
    "1. add / mul / sub",
    lambda a, b: a * b + b - a,
    lambda a, b: a * b + b - a,
    [-4.0, 2.0],
)

compare_to_torch(
    "2. pow and div",
    lambda a, b: (a ** 3) / b + b ** -2,
    lambda a, b: (a ** 3) / b + b ** -2,
    [2.5, 1.7],
)

compare_to_torch(
    "3. tanh",
    lambda a, b: (a * b).tanh() + a.tanh(),
    lambda a, b: torch.tanh(a * b) + torch.tanh(a),
    [0.7, -1.3],
)

compare_to_torch(
    "4. exp",
    lambda a, b: (a + b).exp() * a,
    lambda a, b: torch.exp(a + b) * a,
    [0.4, -0.9],
)

compare_to_torch(
    "5. relu (positive side)",
    lambda a, b: (a * b).relu() + (a + b).relu(),
    lambda a, b: torch.relu(a * b) + torch.relu(a + b),
    [1.5, 2.0],
)

compare_to_torch(
    "6. relu (negative side -- grad should be 0 through it)",
    lambda a, b: (a * b).relu() + a,
    lambda a, b: torch.relu(a * b) + a,
    [-1.5, 2.0],
)

compare_to_torch(
    "7. reverse ops (2 - a, 2 / a, 2 * a)",
    lambda a, b: 2 - a + 2 / b + 2 * a,
    lambda a, b: 2 - a + 2 / b + 2 * a,
    [3.0, 1.5],
)

# ------------------------------------------------- node reuse (the hard one)
compare_to_torch(
    "8. NODE REUSE: a used many times",
    lambda a, b: a * a + a * b + a,
    lambda a, b: a * a + a * b + a,
    [3.0, -2.0],
)

# ------------------------------------------------------- the karpathy gauntlet
def gauntlet_mine(a, b):
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    g = e ** 2
    return g / 2.0 + 10.0 / g

def gauntlet_torch(a, b):
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    g = e ** 2
    return g / 2.0 + 10.0 / g

compare_to_torch("9. THE GAUNTLET (every op, heavy reuse)",
                 gauntlet_mine, gauntlet_torch, [-4.0, 2.0])


# --------------------------------------------------------------- zero_grad
print("\n10. zero_grad")
a = Value(2.0)
o1 = a * 10.0
o1.backward()
check("1st backward", a.grad, 10.0)
a.zero_grad()
o2 = a * 100.0
o2.backward()
check("2nd backward after zero_grad", a.grad, 100.0)

# and prove it breaks WITHOUT the reset
b = Value(2.0)
(b * 10.0).backward()
(b * 100.0).backward()          # no zero_grad on purpose
print(f"  (sanity) no-reset gives {b.grad:+.1f} -- should be 110, i.e. accumulated. "
      f"{'correct accumulation behavior' if abs(b.grad - 110.0) < 1e-9 else 'UNEXPECTED'}")


# --------------------------------------------------- numerical gradient check
print("\n11. numerical gradient check (central difference)")

def grad_check(f, x_val, h=1e-5, tol=1e-4):
    x = Value(x_val)
    y = f(x)
    y.backward()
    analytic = x.grad
    numeric = (f(Value(x_val + h)).data - f(Value(x_val - h)).data) / (2 * h)
    return analytic, numeric, abs(analytic - numeric) < tol

for label, fn, pts in [
    ("tanh", lambda x: x.tanh(),                 [-3, -0.5, 0.0, 0.5, 3]),
    ("exp",  lambda x: x.exp(),                  [-2, -0.5, 0.0, 0.5, 2]),
    ("pow",  lambda x: x ** 3,                   [-2, -0.5, 0.5, 2]),
    ("comp", lambda x: (x.exp() + x).tanh(),     [-1.5, -0.3, 0.3, 1.5]),
]:
    allok = True
    for p in pts:
        a_, n_, ok = grad_check(fn, p)
        if not ok:
            allok = False
            print(f"  FAIL  {label} at x={p}: {a_:.6f} vs {n_:.6f}")
    if allok:
        print(f"  PASS  {label:<28} all {len(pts)} points match")
        passed += 1
    else:
        failed += 1


# ------------------------------------------------------------ MLP training
print("\n12. MLP trains (Karpathy's tiny dataset)")
random.seed(42)
model = MLP(3, [4, 4, 1])
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for step in range(200):
    preds = [model(x) for x in xs]
    loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), start=Value(0.0)) 
    # initially this sum thing was erroring because sum() returns an int not a Value object, so initialized start so that overloaded + is used
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.05 * p.grad

print(f"  final loss: {loss.data:.6f}")
print(f"  predictions: {[round(model(x).data, 3) for x in xs]}")
print(f"  targets:     {ys}")
if loss.data < 0.01:
    print("  PASS  MLP converged")
    passed += 1
else:
    print("  FAIL  MLP did not converge")
    failed += 1


print("\n" + "=" * 60)
print(f"{passed} passed, {failed} failed")
print("=" * 60)
