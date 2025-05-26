# environment/render.py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def render_grid(grid, agent_positions, pause=0.2, step=None, save_dir=None):
    color_map = {0: 'green', 1: 'red', 2: 'saddlebrown'}
    grid_size = grid.shape[0]
    img = np.zeros((grid_size, grid_size, 3))

    for x in range(grid_size):
        for y in range(grid_size):
            c = color_map[grid[x, y]]
            img[x, y, :] = mcolors.to_rgb(c)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, origin='lower')
    # Agent overlays
    for name, pos in agent_positions.items():
        color = {
            'helicopter': 'blue',
            'drone': 'purple',
            'groundcrew': 'black'
        }[name]
        ax.scatter(pos[1], pos[0], c=color, s=120, marker='o', edgecolors='white', linewidths=1.5, label=name, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    print("STEP:", step)
    if save_dir is not None and step is not None:
        #print("Image saving...")
        filename = f"{save_dir}/step_{step:03d}.png"
        plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
