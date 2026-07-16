"""
Franka Panda + Flexible Beam Visualizer
----------------------------------------
Visualizes the Franka Panda robot arm and a flexible beam attached to its
end-effector, driven by two CasADi functions:

    fk_T        : (q[7]) -> (T_joint1..T_joint7, T_ee)   [panda_fk.casadi]
    beam_end_pos: (q[7], th, L) -> (beam_end_pos[3])

Usage:
    python panda_beam_viz.py

Falls back to built-in DH kinematics / straight-beam if CasADi functions
are not available.

Sliders:
    q0–q6  : joint angles (rad)
    th     : beam deflection angle (rad)
    L      : beam length (m)
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if you prefer Qt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patheffects as pe

# ── Load CasADi functions ────────────────────────────────────────────────────
try:
    import casadi as ca

    # ← Point these to your actual .casadi files (or assign ca.Function objects)
    fk_T_fn          = ca.Function.load("panda/panda_fk.casadi")
    beam_end_pos_fn  = ca.Function.load("panda/panda_beam_end_pos.casadi")

    FK_OK     = True
    CASADI_OK = True
    print("[casadi] Loaded panda_fk.casadi  ✓")
    print("[casadi] Loaded beam_end_pos.casadi  ✓")

except Exception as _e:
    print(f"[casadi] Could not load functions ({_e}), using built-in fallbacks.")
    fk_T_fn         = None
    beam_end_pos_fn = None
    FK_OK     = False
    CASADI_OK = False

# ── Optional: roboticstoolbox-python ────────────────────────────────────────
# Install with:  pip install roboticstoolbox-python
# Used ONLY for authoritative joint-frame positions via fkine_all().
# We never call robot.plot() — rendering is always done by our own code.
try:
    import roboticstoolbox as rtb
    RTB_OK = True
    _panda_model = rtb.models.Panda()
    print("[rtb] roboticstoolbox Panda model loaded  ✓")
except Exception as _rtb_e:
    RTB_OK = False
    _panda_model = None
    print(f"[rtb] roboticstoolbox not available ({_rtb_e}), using built-in DH fallback.")


# Columns: [a,   d,      alpha,       theta_offset]
PANDA_DH = np.array([
    [0,      0.333,   0,          0],
    [0,      0,      -np.pi/2,   0],
    [0,      0.316,   np.pi/2,   0],
    [0.0825, 0,       np.pi/2,   0],
    [-0.0825,0.384,  -np.pi/2,   0],
    [0,      0,       np.pi/2,   0],
    [0.088,  0,       np.pi/2,   0],
])
# Flange → end-effector translation
EE_OFFSET = np.array([0, 0, 0.107])

# Joint limits [min, max] in radians
JOINT_LIMITS = [
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
]

# ── Link colours (proximal → distal) ────────────────────────────────────────
LINK_COLORS = [
    "#4E9AF1",   # 0-1  base → 1
    "#4E9AF1",   # 1-2
    "#3B7DD8",   # 2-3
    "#3B7DD8",   # 3-4
    "#2A5FAA",   # 4-5
    "#2A5FAA",   # 5-6
    "#1B3F72",   # 6-EE
    "#1B3F72",   # 6-EE
]
JOINT_COLOR   = "#E0E0E0"
EE_COLOR      = "#FF6B35"
# BEAM_COLOR    = "#27AE60"
BEAM_COLOR_NORMAL = [0.2, 0.8, 0.2]
BEAN_COLOR_FLEX = [0.8, 0.2, 0.2]
BEAM_END_COLOR= "#E74C3C"


# ── Kinematics helpers ───────────────────────────────────────────────────────

axes_limits = {
    "x": (-0.3, 0.7),
    "y": (-0.5, 0.5),
    "z": (0.0, 1.0),
}

def dh_transform(a, d, alpha, theta):
    """Modified DH transformation matrix."""
    ca_, sa_ = np.cos(alpha), np.sin(alpha)
    ct, st   = np.cos(theta), np.sin(theta)
    return np.array([
        [ ct,  -st,   0,   a],
        [ st*ca_, ct*ca_, -sa_, -sa_*d],
        [ st*sa_, ct*sa_,  ca_,  ca_*d],
        [  0,    0,    0,   1],
    ])


def forward_kinematics(q):
    """
    Returns list of 9 × (4×4) np.ndarray transforms:
        [world, T_joint1 .. T_joint7, T_ee]

    Priority:
      1. panda_fk.casadi  (your function, most accurate)
      2. roboticstoolbox  fkine_all()  (authoritative DH)
      3. built-in DH      (pure-numpy fallback)
    """
    frames_DM = fk_T_fn(q)
    frames = [np.eye(4)]
    for i in range(len(frames_DM)):
        frames.append(np.array(frames_DM[i], dtype=float).reshape(4, 4))
    return frames
    # if FK_OK:
    #     results = fk_T_fn(q)           # 8 DM outputs, each 4×4
    #     frames  = [np.eye(4)]
    #     for T_dm in results:
    #         frames.append(np.array(T_dm, dtype=float).reshape(4, 4))
    #     return frames                  # 9 frames

    # if RTB_OK:
    #     # fkine_all returns an SE3 with one pose per link (including base)
    #     # shape: (n_links+1,) — index 0 is base, last is EE
    #     traj = _panda_model.fkine_all(q)   # SE3 of length 9 (base + 7 joints + EE)
    #     frames = []
    #     for i in range(len(traj)):
    #         frames.append(np.array(traj[i].A, dtype=float))
    #     # fkine_all already includes base (identity) and EE, so length == 9
    #     if len(frames) == 8:            # some rtb versions omit base frame
    #         frames.insert(0, np.eye(4))
    #     return frames                  # 9 frames

    # # ── Pure-numpy DH fallback ───────────────────────────────────────────────
    # T = np.eye(4)
    # frames = [T.copy()]
    # for i, (a, d, alpha, theta_off) in enumerate(PANDA_DH):
    #     T = T @ dh_transform(a, d, alpha, q[i] + theta_off)
    #     frames.append(T.copy())
    # T_ee = frames[-1].copy()
    # T_ee[:3, 3] += T_ee[:3, :3] @ EE_OFFSET
    # frames.append(T_ee)
    # return frames


def beam_endpoint_fallback(T_ee, th, L):
    """
    Fallback when CasADi is unavailable.
    Models the beam as deflecting in the EE local XZ plane by angle th.
    """
    # beam direction in EE local frame
    local_dir = np.array([np.sin(th), 0, np.cos(th)])
    world_dir = T_ee[:3, :3] @ local_dir
    return T_ee[:3, 3] + L * world_dir


def beam_endpoint(q, th, L, T_ee):
    """Compute beam end position, using CasADi fn if available."""
    if CASADI_OK:
        res = beam_end_pos_fn(q, th, L)
        return np.array(res, dtype=float).flatten()
    return beam_endpoint_fallback(T_ee, th, L)


# ── Drawing helpers ──────────────────────────────────────────────────────────

def draw_cylinder_link(ax, p0, p1, radius=0.028, color="#4E9AF1", alpha=0.85, n=12):
    """Draw a thick 3-D cylinder between two points."""
    v   = p1 - p0
    L   = np.linalg.norm(v)
    if L < 1e-6:
        return
    v_hat = v / L

    # build an orthonormal frame around v_hat
    perp = np.array([1, 0, 0]) if abs(v_hat[0]) < 0.9 else np.array([0, 1, 0])
    u    = np.cross(v_hat, perp);  u /= np.linalg.norm(u)
    w    = np.cross(v_hat, u)

    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle = radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), w))

    verts_bot = p0 + circle          # (n, 3)
    verts_top = p1 + circle

    # side faces
    for i in range(n):
        j = (i + 1) % n
        xs = [verts_bot[i,0], verts_bot[j,0], verts_top[j,0], verts_top[i,0]]
        ys = [verts_bot[i,1], verts_bot[j,1], verts_top[j,1], verts_top[i,1]]
        zs = [verts_bot[i,2], verts_bot[j,2], verts_top[j,2], verts_top[i,2]]
        ax.plot_surface(
            np.array([xs[:2], xs[2:][::-1]]),
            np.array([ys[:2], ys[2:][::-1]]),
            np.array([zs[:2], zs[2:][::-1]]),
            color=color, alpha=alpha, shade=True, linewidth=0
        )


def draw_sphere(ax, centre, radius=0.04, color=JOINT_COLOR, alpha=0.9):
    u = np.linspace(0, 2*np.pi, 14)
    v = np.linspace(0,   np.pi, 10)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True, linewidth=0)


def draw_beam(ax, p_ee, p_end, n_pts=40, th=0.0):
    """Draw a curved beam from EE to beam endpoint (catenary-ish)."""
    # Ensure float64 — CasADi DM arrays can come through as integer dtype
    p_ee  = np.asarray(p_ee,  dtype=float).flatten()
    p_end = np.asarray(p_end, dtype=float).flatten()
    # simple quadratic sag for visual appeal
    t  = np.linspace(0, 1, n_pts)
    pts = np.outer(1-t, p_ee) + np.outer(t, p_end)
    # add a small sag perpendicular to beam axis (gravity proxy)
    sag_dir = np.array([0.0, 0.0, -1.0])
    sag_dir -= sag_dir.dot(p_end - p_ee) / (np.linalg.norm(p_end-p_ee)**2 + 1e-9) * (p_end-p_ee)
    sag_norm = np.linalg.norm(sag_dir)
    if sag_norm > 1e-6:
        sag_dir /= sag_norm
    beam_len = np.linalg.norm(p_end - p_ee)
    sag_mag  = 0.03 * beam_len * np.sin(abs(th)) if abs(th) < np.pi/2 else 0
    sag      = sag_mag * 4 * t * (1 - t)
    pts     += np.outer(sag, sag_dir)

    max_flex = 0.1
    flex_color = np.array(BEAM_COLOR_NORMAL) * (1 - min(abs(th)/max_flex, 1.0)) + np.array(BEAN_COLOR_FLEX) * min(abs(th)/max_flex, 1.0)

    ax.plot(pts[:,0], pts[:,1], pts[:,2],
            color=flex_color, linewidth=3.5, zorder=5,
            solid_capstyle="round")
    # endpoint marker
    ax.scatter(*p_end, color=BEAM_END_COLOR, s=80, zorder=6, depthshade=False)


def draw_frame_axes(ax, T, scale=0.06):
    """Draw a small RGB XYZ frame at transform T."""
    o  = T[:3, 3]
    for i, c in enumerate(["#e74c3c", "#2ecc71", "#3498db"]):
        d = T[:3, i]
        ax.quiver(*o, *d*scale, color=c, linewidth=1.2, arrow_length_ratio=0.3)


# ── Main drawing function ────────────────────────────────────────────────────

# Link radii (visual, not real)
LINK_RADII = [0.040, 0.038, 0.036, 0.032, 0.028, 0.026, 0.022, 0.022]


def show_robot_with_beam(ax, q, th, L):
    """
    Plot the Franka Panda robot at joint configuration q[7], with the
    flexible beam (deflection angle th, length L) attached to the EE.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
    q  : array-like, shape (7,) – joint angles in radians
    th : float – beam deflection angle (rad)
    L  : float – beam length (m)
    """
    ax.cla()
    q = np.asarray(q, dtype=float)

    frames = forward_kinematics(q)
    joint_positions = [np.asarray(f[:3, 3], dtype=float) for f in frames]
    T_ee  = frames[8]
    p_ee  = joint_positions[8]
    p_end = beam_endpoint(q, th, L, T_ee)

    _draw_robot_cylinders(ax, joint_positions)

    # ── Draw beam ───────────────────────────────────────────────────────────
    draw_beam(ax, p_ee, p_end, th=th)

    # EE frame axes
    draw_frame_axes(ax, T_ee, scale=0.07)

    # ── Legend ──────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=LINK_COLORS[0],  label="Panda links"),
        mpatches.Patch(color=JOINT_COLOR,      label="Joints"),
        mpatches.Patch(color=EE_COLOR,         label="End-effector"),
        # mpatches.Patch(color=BEAM_COLOR,       label="Beam"),
        mpatches.Patch(color=BEAM_END_COLOR,   label="Beam tip"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              framealpha=0.6, handlelength=1.2)

    # ── Axes cosmetics ───────────────────────────────────────────────────────
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.set_title("Franka Panda + Flexible Beam", fontsize=10, fontweight="bold")

    # Equal aspect: find the largest range and centre all axes around it
    all_pts = np.array(joint_positions + [p_end], dtype=float)
    lo = all_pts.min(axis=0) - 0.1
    hi = all_pts.max(axis=0) + 0.1
    centre = (lo + hi) / 2
    half   = max(hi - lo) / 2          # same half-range for all axes

    # ax.set_xlim(centre[0] - half, centre[0] + half)
    # ax.set_ylim(centre[1] - half, centre[1] + half)
    # ax.set_zlim(max(0, centre[2] - half), centre[2] + half)
    ax.set_xlim(*axes_limits["x"])
    ax.set_ylim(*axes_limits["y"])
    ax.set_zlim(*axes_limits["z"])
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

def animate_stationary_robot_with_beam(ax, q, th0, dth0, wn, zeta, L):
    """Animate the beam deflecting while the robot stays stationary."""
    import matplotlib.animation as animation

    ddtheta_func = ca.Function.load("panda/panda_beam_model.casadi")

    # simulate
    th_vals = [th0]
    thd = 0.0
    thdd = 0.0
    dt = 0.01
    for i in range(1000):
        thdd = ddtheta_func(q, np.zeros(7), np.zeros(7), wn, zeta, L, th_vals[-1], thd).full().item()
        thd += thdd * dt
        th_vals.append(th_vals[-1] + thd * dt)

    def update(frame):
        th = th_vals[frame]
        show_robot_with_beam(ax, q, th, L)

    ani = animation.FuncAnimation(ax.figure, update, frames=len(th_vals), interval=50)
    # plt.show()
    return ani

def show_robot_with_surface(ax, q, floor_height):
    """
    Plot the Franka Panda robot at joint configuration q[7]

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
    q  : array-like, shape (7,) – joint angles in radians
    """
    ax.cla()
    # ── Draw surface ───────────────────────────────────────────────────────────
    x = np.linspace(0.1, 0.5, 20)
    y = np.linspace(-0.2, 0.4, 20)
    x, y = np.meshgrid(x, y)
    z = np.full_like(x, floor_height)
    ax.plot_surface(x, y, z, color="#27ae60", alpha=0.2, zorder=-100)

    q = np.asarray(q, dtype=float)

    frames = forward_kinematics(q)
    joint_positions = [np.asarray(f[:3, 3], dtype=float) for f in frames]
    T_ee  = frames[8]
    p_ee  = joint_positions[8]

    _draw_robot_cylinders(ax, joint_positions)


    # EE frame axes
    draw_frame_axes(ax, T_ee, scale=0.07)

    # ── Legend ──────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=LINK_COLORS[0],  label="Panda links"),
        mpatches.Patch(color=JOINT_COLOR,      label="Joints"),
        mpatches.Patch(color=EE_COLOR,         label="End-effector"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              framealpha=0.6, handlelength=1.2)

    # ── Axes cosmetics ───────────────────────────────────────────────────────
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.set_title("Franka Panda", fontsize=10, fontweight="bold")
    ax.set_xlim(*axes_limits["x"])
    ax.set_ylim(*axes_limits["y"])
    ax.set_zlim(*axes_limits["z"])
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

def _draw_robot_cylinders(ax, joint_positions):
    """Fallback: draw links as cylinders and joints as spheres."""
    # base plate
    th_c = np.linspace(0, 2*np.pi, 40)
    bx, by = 0.08 * np.cos(th_c), 0.08 * np.sin(th_c)
    bz = np.zeros_like(bx)
    ax.plot_surface(np.array([bx, bx]), np.array([by, by]),
                    np.array([bz - 0.015, bz]),
                    color="#555", alpha=0.9, linewidth=0)

    for i in range(8):
        p0 = joint_positions[i]
        p1 = joint_positions[i + 1]
        draw_cylinder_link(ax, p0, p1, radius=LINK_RADII[i], color=LINK_COLORS[i])

    for i in range(1, 8):
        draw_sphere(ax, joint_positions[i],
                    radius=LINK_RADII[min(i-1, 6)] * 1.15, color=JOINT_COLOR)

    # EE flange
    draw_sphere(ax, joint_positions[8], radius=0.025, color=EE_COLOR)

# ── Interactive GUI ──────────────────────────────────────────────────────────

def launch_gui():
    # Default configuration: Panda "home" pose
    q_init = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
    th_init = 0.3
    L_init  = 0.4

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#1a1a2e")

    ax3d = fig.add_axes([0.02, 0.22, 0.65, 0.75], projection="3d")
    ax3d.set_facecolor("#1a1a2e")

    # ── Sliders layout ───────────────────────────────────────────────────────
    slider_axes = []
    sliders     = []

    n_sliders = 9   # q0..q6, th, L
    left, width  = 0.70, 0.27
    bottom_start = 0.88
    height, gap  = 0.030, 0.010

    labels  = [f"q{i} (rad)" for i in range(7)] + ["θ (rad)", "L (m)"]
    limits  = JOINT_LIMITS + [(-np.pi/2, np.pi/2), (0.05, 1.0)]
    inits   = list(q_init) + [th_init, L_init]

    for i in range(n_sliders):
        b = bottom_start - i * (height + gap)
        sa = fig.add_axes([left, b, width, height])
        sa.set_facecolor("#2a2a4a")
        lo, hi = limits[i]
        sl = Slider(sa, labels[i], lo, hi, valinit=inits[i],
                    color="#4E9AF1", track_color="#2a2a4a")
        sl.label.set_color("white")
        sl.label.set_fontsize(8)
        sl.valtext.set_color("#FF6B35")
        sl.valtext.set_fontsize(8)
        slider_axes.append(sa)
        sliders.append(sl)

    # reset button
    from matplotlib.widgets import Button
    btn_ax = fig.add_axes([left + 0.06, 0.03, 0.10, 0.04])
    btn    = Button(btn_ax, "Reset", color="#2a2a4a", hovercolor="#4E9AF1")
    btn.label.set_color("white")

    # info text
    info_ax = fig.add_axes([0.02, 0.02, 0.62, 0.18])
    info_ax.set_facecolor("#12122a")
    info_ax.axis("off")
    info_text = info_ax.text(
        0.01, 0.95, "", transform=info_ax.transAxes,
        fontsize=8, color="#aaaaff", va="top", family="monospace"
    )

    def get_values():
        q  = np.array([sliders[i].val for i in range(7)])
        th = sliders[7].val
        L  = sliders[8].val
        return q, th, L

    def update_info(q, th, L, p_end):
        frames = forward_kinematics(q)
        p_ee   = frames[8][:3, 3]
        if FK_OK:
            fk_status = "✓ panda_fk.casadi"
        elif RTB_OK:
            fk_status = "✓ rtb fkine_all"
        else:
            fk_status = "✗ built-in DH"
        beam_status = "✓ beam_end_pos.casadi" if CASADI_OK else "✗ geometric fallback"
        lines  = [
            f"  EE position  : [{p_ee[0]:+.4f}, {p_ee[1]:+.4f}, {p_ee[2]:+.4f}] m",
            f"  Beam tip     : [{p_end[0]:+.4f}, {p_end[1]:+.4f}, {p_end[2]:+.4f}] m",
            f"  Beam length  : {L:.3f} m   |   θ : {th:.3f} rad ({np.degrees(th):.1f}°)",
            f"  FK : {fk_status}   |   Beam : {beam_status}",
        ]
        info_text.set_text("\n".join(lines))

    def redraw(_=None):
        q, th, L = get_values()
        show_robot_with_beam(ax3d, q, th, L)
        frames = forward_kinematics(q)
        T_ee   = frames[8]
        p_end  = beam_endpoint(q, th, L, T_ee)
        update_info(q, th, L, p_end)
        fig.canvas.draw_idle()

    def reset(_):
        for i, sl in enumerate(sliders):
            sl.set_val(inits[i])

    for sl in sliders:
        sl.on_changed(redraw)
    btn.on_clicked(reset)

    redraw()
    plt.show()

if __name__ == "__main__":
    launch_gui()