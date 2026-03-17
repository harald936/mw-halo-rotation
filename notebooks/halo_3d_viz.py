"""
3D visualization of a triaxial, tilted, tumbling dark matter halo.
Exaggerated parameters so you can see the effects clearly.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# --- Exaggerated halo parameters ---
# Real values would be subtle; these are cranked up so you can SEE it
q_x = 1.0    # x-axis ratio (normalized)
q_y = 0.7    # y-axis ratio — makes it triaxial (real: ~0.9)
q_z = 0.5    # z-axis ratio — oblate flattening (real: ~0.93)
tilt_deg = 30  # tilt angle from disk z-axis (real: ~18 deg from Nibauer+Bonaca)

# --- Build ellipsoid surface ---
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 40)
r_halo = 15  # scale for visualization (kpc)

# Triaxial ellipsoid: different radii along x, y, z
x = r_halo * q_x * np.outer(np.cos(u), np.sin(v))
y = r_halo * q_y * np.outer(np.sin(u), np.sin(v))
z = r_halo * q_z * np.outer(np.ones_like(u), np.cos(v))

# --- Apply tilt (rotate around y-axis) ---
tilt = np.radians(tilt_deg)
x_tilt = x * np.cos(tilt) + z * np.sin(tilt)
y_tilt = y
z_tilt = -x * np.sin(tilt) + z * np.cos(tilt)

# --- Build disk (thin flat circle) ---
disk_r = np.linspace(0, 20, 30)
disk_theta = np.linspace(0, 2 * np.pi, 60)
DR, DT = np.meshgrid(disk_r, disk_theta)
disk_x = DR * np.cos(DT)
disk_y = DR * np.sin(DT)
disk_z = np.zeros_like(disk_x)

# --- Figure with 3 panels showing tumbling ---
fig = plt.figure(figsize=(20, 7))

omega_p = 0.1  # km/s/kpc — tumbling rate
times_gyr = [0, 3, 6]  # Gyr — show halo at different times
rotation_angles = [omega_p * t * 1.0226 for t in times_gyr]  # convert to radians (approx)
# Exaggerate rotation for visibility
rotation_angles = [0, np.pi/4, np.pi/2]  # 0, 45, 90 degrees

for panel, (rot_angle, t_label) in enumerate(zip(rotation_angles, ["t = 0 (now)", "t = -3 Gyr", "t = -6 Gyr"])):
    ax = fig.add_subplot(1, 3, panel + 1, projection='3d')

    # Rotate halo around disk z-axis (this is the tumbling)
    cos_r = np.cos(rot_angle)
    sin_r = np.sin(rot_angle)
    x_rot = x_tilt * cos_r - y_tilt * sin_r
    y_rot = x_tilt * sin_r + y_tilt * cos_r
    z_rot = z_tilt

    # Plot disk
    ax.plot_surface(disk_x, disk_y, disk_z, alpha=0.15, color='royalblue',
                    label='Galactic disk')

    # Plot halo (transparent ellipsoid)
    ax.plot_surface(x_rot, y_rot, z_rot, alpha=0.25, color='darkviolet',
                    edgecolor='purple', linewidth=0.1)

    # Plot halo major axis
    major_end = np.array([r_halo * 1.3, 0, 0])
    # Apply tilt
    major_tilt = np.array([
        major_end[0] * np.cos(tilt) + major_end[2] * np.sin(tilt),
        major_end[1],
        -major_end[0] * np.sin(tilt) + major_end[2] * np.cos(tilt)
    ])
    # Apply rotation
    major_rot = np.array([
        major_tilt[0] * cos_r - major_tilt[1] * sin_r,
        major_tilt[0] * sin_r + major_tilt[1] * cos_r,
        major_tilt[2]
    ])
    ax.plot([0, major_rot[0]], [0, major_rot[1]], [0, major_rot[2]],
            'r-', linewidth=3, label='Halo major axis')
    ax.plot([0, -major_rot[0]], [0, -major_rot[1]], [0, -major_rot[2]],
            'r-', linewidth=3)

    # Plot z-axis (disk normal)
    ax.plot([0, 0], [0, 0], [0, 18], 'b--', linewidth=1.5, alpha=0.5, label='Disk z-axis')

    # Plot halo minor axis (the short axis)
    minor_end = np.array([0, 0, r_halo * q_z * 1.3])
    minor_tilt = np.array([
        minor_end[0] * np.cos(tilt) + minor_end[2] * np.sin(tilt),
        minor_end[1],
        -minor_end[0] * np.sin(tilt) + minor_end[2] * np.cos(tilt)
    ])
    minor_rot = np.array([
        minor_tilt[0] * cos_r - minor_tilt[1] * sin_r,
        minor_tilt[0] * sin_r + minor_tilt[1] * cos_r,
        minor_tilt[2]
    ])
    ax.plot([0, minor_rot[0]], [0, minor_rot[1]], [0, minor_rot[2]],
            'g-', linewidth=2, label='Halo minor axis')

    # Add rotation arrow (curved arrow around z)
    if panel > 0:
        arrow_t = np.linspace(0, rot_angle, 30)
        arrow_r = 18
        ax.plot(arrow_r * np.cos(arrow_t), arrow_r * np.sin(arrow_t),
                np.ones_like(arrow_t) * 12, 'orange', linewidth=2.5)
        # Arrowhead
        ax.scatter([arrow_r * np.cos(rot_angle)], [arrow_r * np.sin(rot_angle)],
                   [12], color='orange', s=80, marker='>')

    # GD-1 orbit (approximate — a tilted circle at ~15 kpc)
    stream_t = np.linspace(-0.8, 0.8, 100)
    stream_r = 14
    stream_x = stream_r * np.cos(stream_t + 0.5)
    stream_y = stream_r * np.sin(stream_t + 0.5)
    stream_z = 8 * np.sin(stream_t)  # GD-1 goes above/below the plane
    ax.plot(stream_x, stream_y, stream_z, 'yellow', linewidth=3, alpha=0.8,
            label='GD-1 stream (approx)')

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-15, 15)
    ax.set_xlabel('X (kpc)')
    ax.set_ylabel('Y (kpc)')
    ax.set_zlabel('Z (kpc)')
    ax.set_title(t_label, fontsize=14, fontweight='bold')
    ax.view_init(elev=25, azim=45 - panel * 15)

    if panel == 0:
        ax.legend(fontsize=7, loc='upper left')

plt.suptitle(
    'Triaxial Tilted Tumbling Dark Matter Halo (EXAGGERATED)\n'
    f'q_x={q_x}, q_y={q_y}, q_z={q_z}, tilt={tilt_deg}°  |  '
    'Purple = halo, Blue = disk, Yellow = GD-1, Red = halo major axis\n'
    'The halo rotates around the disk z-axis → GD-1 feels a different gravitational field at each time',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('/Users/haralds./mw-halo-rotation/results/plots/halo_3d_tumbling.png', dpi=200)
plt.show()
print("Saved to results/plots/halo_3d_tumbling.png")
