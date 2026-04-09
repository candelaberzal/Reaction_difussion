import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os

# ============================================================
# PDE
# ============================================================

def gierer_meinhardt_pde(uv, a=0.4, b=1.0, d=50, dx=0.5):

    lap = -4 * uv
    lap += np.roll(uv, 1, axis=1)
    lap += np.roll(uv, -1, axis=1)
    lap += np.roll(uv, 1, axis=2)
    lap += np.roll(uv, -1, axis=2)
    lap /= dx**2

    u, v = uv
    lu, lv = lap

    f = a - b*u + u**2 / (v + 1e-8)
    g = u**2 - v

    du = lu + f
    dv = d * lv + g

    return np.array([du, dv])


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(d_value):

    dx = 0.5
    dt = 0.001

    L = 40
    N = int(L/dx)

    steps = 40000
    save_every = 200

    a, b = 0.4, 1.0

    # steady state
    u_star = (a + 1)/b
    v_star = u_star**2

    uv = np.zeros((2, N, N))

    noise = np.random.randn(N, N)

    # smooth noise (important)
    noise = (noise +
             np.roll(noise,1,0) + np.roll(noise,-1,0) +
             np.roll(noise,1,1) + np.roll(noise,-1,1)) / 5

    uv[0] = u_star * (1 + 0.01*noise)
    uv[1] = v_star * (1 + 0.01*noise)

    snapshots = []

    for t in range(steps):

        uv += dt * gierer_meinhardt_pde(uv, d=d_value, dx=dx)

        # Neumann BC
        uv[:, 0, :] = uv[:, 1, :]
        uv[:, -1, :] = uv[:, -2, :]
        uv[:, :, 0] = uv[:, :, 1]
        uv[:, :, -1] = uv[:, :, -2]

        if t % save_every == 0:
            snapshots.append(uv[1].copy())

    return snapshots


# ============================================================
# SAVE SNAPSHOT IMAGE
# ============================================================

def save_snapshot_grid(snapshots, title, folder):

    fig, axs = plt.subplots(2, 3, figsize=(10,6))

    idx = np.linspace(0, len(snapshots)-1, 6).astype(int)

    for ax, i in zip(axs.flat, idx):
        ax.imshow(snapshots[i], cmap="viridis")
        ax.set_title(f"t={i*200}")
        ax.axis("off")

    plt.suptitle(title)

    path = os.path.join(folder, f"{title}_snapshots.png")
    print("Saving snapshot:", path)
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
# SAVE GIF
# ============================================================

def save_gif(snapshots, title, folder):

    fig, ax = plt.subplots()
    im = ax.imshow(snapshots[0], cmap="viridis")

    def update(frame):
        im.set_array(snapshots[frame])
        ax.set_title(f"{title} | t={frame*200}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots))

    path = os.path.join(folder, f"{title}.gif")
    print("Saving GIF:", path)

    ani.save(path, writer=PillowWriter(fps=20))
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # 📁 save inside project folder
    base_path = os.path.dirname(__file__)
    save_folder = os.path.join(base_path, "gm2d_outputs")
    os.makedirs(save_folder, exist_ok=True)

    # -----------------------------
    # CASE 1: UNSTABLE
    # -----------------------------
    print("\nRunning UNSTABLE case (d = 50)...")
    snaps_unstable = run_simulation(d_value=50)

    save_snapshot_grid(snaps_unstable, "GM2D_unstable", save_folder)
    save_gif(snaps_unstable, "GM2D_unstable", save_folder)

    # -----------------------------
    # CASE 2: STABLE
    # -----------------------------
    print("\nRunning STABLE case (d = 20)...")
    snaps_stable = run_simulation(d_value=20)

    save_snapshot_grid(snaps_stable, "GM2D_stable", save_folder)
    save_gif(snaps_stable, "GM2D_stable", save_folder)