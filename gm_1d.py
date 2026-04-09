import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os

# ============================================================
# PDE
# ============================================================

def gierer_meinhardt_pde(uv, a=0.4, b=1.0, d=30, dx=1.0):
    lap = -2 * uv
    lap += np.roll(uv, 1, axis=1)
    lap += np.roll(uv, -1, axis=1)
    lap /= dx**2

    u, v = uv
    lu, lv = lap

    f = a - b*u + u**2 / (v + 1e-8)
    g = u**2 - v

    du_dt = lu + f
    dv_dt = d * lv + g

    return np.array([du_dt, dv_dt])


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(a=0.4, b=1.0, d=30):
    N = 40
    dx = 1.0
    dt = 0.01
    steps = 50000
    save_every = 500

    u_star = (a + 1) / b
    v_star = u_star**2

    uv = np.zeros((2, N))
    uv[0] = u_star * (1 + 0.01 * np.random.randn(N))
    uv[1] = v_star * (1 + 0.01 * np.random.randn(N))

    snapshots = []

    for t in range(steps):
        uv += dt * gierer_meinhardt_pde(uv, a, b, d, dx)

        uv[:, 0] = uv[:, 1]
        uv[:, -1] = uv[:, -2]

        if t % save_every == 0:
            snapshots.append(uv.copy())

    return snapshots


# ============================================================
# SAVE SNAPSHOT IMAGE
# ============================================================

def save_snapshots_1d(snapshots, title, folder):

    plt.figure(figsize=(8,5))
    idx = np.linspace(len(snapshots)//3, len(snapshots)-1, 6).astype(int)

    for i in idx:
        v = snapshots[i][1]
        plt.plot(v, label=f"t={i*500}")

    plt.xlabel("x")
    plt.ylabel("v(x)")
    plt.title(title)
    plt.legend()

    path = os.path.join(folder, f"{title}_snapshots.png")
    print("Saving:", path)
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
# SAVE GIF
# ============================================================

def save_gif(snapshots, title, folder):

    fig, ax = plt.subplots()
    line, = ax.plot(snapshots[0][1], lw=2)

    def update(frame):
        line.set_ydata(snapshots[frame][1])
        ax.set_title(f"{title} | t={frame*500}")
        return [line]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots))

    path = os.path.join(folder, f"{title}.gif")
    print("Saving GIF:", path)

    ani.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)


# ============================================================
# TURING SPACE (FIXED)
# ============================================================

def turing_space_plot(folder):

    a_vals = np.linspace(0.01, 1.0, 300)
    d_vals = np.linspace(0.1, 100, 300)

    A, D = np.meshgrid(a_vals, d_vals)

    b = 1.0

    fu = 2*b/(A+1) - b
    fv = -(b/(A+1))**2
    gu = 2*(A+1)/b
    gv = -1.0

    Delta = fu*gv - fv*gu

    # 🔥 CRITICAL FIX
    Delta = np.maximum(Delta, 1e-8)

    cond1 = (fu + gv) < 0
    cond2 = Delta > 0
    cond3 = (gv + D*fu) > (2*np.sqrt(D)*np.sqrt(Delta))

    mask = cond1 & cond2 & cond3

    plt.figure(figsize=(7,5))
    plt.contourf(A, D, mask, levels=[-0.1, 0.5, 1], cmap="viridis")

    plt.xlabel("a")
    plt.ylabel("d")
    plt.title("Turing space (b = 1)")

    plt.scatter([0.4, 0.4], [20, 30], color="red")
    plt.text(0.42, 30, "d=30")
    plt.text(0.42, 20, "d=20")

    path = os.path.join(folder, "turing_space.png")
    print("Saving:", path)
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    base_path = os.path.dirname(__file__)
    folder = os.path.join(base_path, "gm1d_outputs")
    os.makedirs(folder, exist_ok=True)

    # ------------------------
    # UNSTABLE CASE
    # ------------------------
    snaps30 = run_simulation(d=30)
    save_snapshots_1d(snaps30, "GM1D_unstable", folder)
    save_gif(snaps30, "GM1D_unstable", folder)

    # ------------------------
    # STABLE CASE
    # ------------------------
    snaps20 = run_simulation(d=20)
    save_snapshots_1d(snaps20, "GM1D_stable", folder)
    save_gif(snaps20, "GM1D_stable", folder)

    # ------------------------
    # TURING SPACE
    # ------------------------
    turing_space_plot(folder)