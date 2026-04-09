import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os

# ============================================================
# LAPLACIAN (CORRECT)
# ============================================================

def laplacian(uv):
    lap = -4 * uv
    lap += np.roll(uv, 1, axis=1)
    lap += np.roll(uv, -1, axis=1)
    lap += np.roll(uv, 1, axis=2)
    lap += np.roll(uv, -1, axis=2)
    return lap


# ============================================================
# PDE
# ============================================================

def gray_scott(uv, D1, D2, F, k):
    u, v = uv
    lu, lv = laplacian(uv)

    uv2 = u * v * v

    du = D1 * lu - uv2 + F * (1 - u)
    dv = D2 * lv + uv2 - (F + k) * v

    return np.array([du, dv])


# ============================================================
# INITIAL CONDITIONS (FIXED FOR ALL TYPES)
# ============================================================

def initial_condition(N, label):
    uv = np.ones((2, N, N))
    uv[1] = 0

    r = 8
    c = N//2

    if label in ["Type A", "Type B", "Type E"]:
        uv[0, c-r:c+r, c-r:c+r] = 0.25
        uv[1, c-r:c+r, c-r:c+r] = 0.5

    elif label == "Type C":
        uv[1] = 0.02 * np.random.rand(N, N)

    elif label == "Type D":
        seeds = [(c,c),(c-40,c-40),(c+40,c+40)]
        for cx, cy in seeds:
            uv[0, cx-r:cx+r, cy-r:cy+r] = 0.3
            uv[1, cx-r:cx+r, cy-r:cy+r] = 0.7

    return uv


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(F, k, label):

    N = 250
    dt = 2
    steps = 80000
    save_every = 400

    D1, D2 = 0.1, 0.05

    uv = initial_condition(N, label)
    snapshots = []

    for t in range(steps):
        uv += dt * gray_scott(uv, D1, D2, F, k)

        if t % save_every == 0:
            snapshots.append(uv[1].copy())

    return snapshots


# ============================================================
# GIF SAVE
# ============================================================

def save_gif(snapshots, title):

    base_path = os.path.dirname(__file__)
    folder = os.path.join(base_path, "gifs")
    os.makedirs(folder, exist_ok=True)

    fig, ax = plt.subplots()
    im = ax.imshow(snapshots[0], cmap="inferno")

    def update(frame):
        im.set_array(snapshots[frame])
        ax.set_title(f"{title} | t={frame*400}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots))

    path = os.path.join(folder, f"{title}.gif")
    print("Saving:", path)
    ani.save(path, writer=PillowWriter(fps=20))
    plt.close()


# ============================================================
# RUN ALL CASES
# ============================================================

if __name__ == "__main__":

    cases = {
        "Type_A": (0.040, 0.060),
        "Type_B": (0.014, 0.047),
        "Type_C": (0.062, 0.065),
        "Type_D": (0.078, 0.061),
        "Type_E": (0.082, 0.059),
    }

    for name, (F, k) in cases.items():
        print("Running", name)
        snaps = run_simulation(F, k, name.replace("_", " "))
        save_gif(snaps, name)