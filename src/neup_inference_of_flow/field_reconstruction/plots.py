import numpy as np

from matplotlib.colors import ListedColormap

CUSTOM_32_CMAP = ListedColormap(
    [
        "#1C1D8A", "#1A2988", "#173386", "#153B84",
        "#124282", "#164E8F", "#1B5A9E", "#2064AC",
        "#246DB8", "#2776C4", "#408BC8", "#519ECB",
        "#60AECF", "#75BEC7", "#A3D27F", "#DBEC62",
        "#FCF957", "#FDED55", "#FDE052", "#FDD24F",
        "#FEC34C", "#FEB349", "#FEA246", "#FF8E43",
        "#FE7C3F", "#FB703A", "#F76334", "#F4542E",
        "#F14127", "#D83820", "#B4331A", "#882E10",
    ],
    name="custom_32",
)


def _per_level_grid(num_levels):
    if not (1 <= num_levels <= 9):
        raise ValueError(f"Expected number of levels between 1 and 9, got {num_levels}.")
    if num_levels == 1:
        return 1, 1
    if num_levels == 2:
        return 1, 2
    if num_levels <= 4:
        return 2, 2
    if num_levels <= 6:
        return 2, 3
    return 3, 3


def _resolve_level_titles(num_levels, titles):
    default_titles = [f"Level {i + 1}" for i in range(num_levels)]
    if titles is None:
        return default_titles
    titles = list(titles)
    return titles[:num_levels] + default_titles[len(titles):]



def plot_mask_composite(geometry_mask, observed_mask, missing_mask, figsize=(9, 8)):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    gm = np.asarray(geometry_mask).astype(bool)
    obs = np.asarray(observed_mask).astype(bool)
    miss = np.asarray(missing_mask).astype(bool)
    comp = np.zeros(gm.shape, dtype=int)
    comp[obs] = 1
    comp[miss] = 2
    cmap = ListedColormap(["white", "royalblue", "crimson"])

    fig, ax = plt.subplots(2, 2, figsize=figsize)
    ax[0, 0].imshow(gm.astype(int), cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1, interpolation="nearest")
    ax[0, 0].set_title(f"Geometry (valid={gm.sum()})")
    ax[0, 1].imshow(miss.astype(int), cmap=ListedColormap(["white", "crimson"]), vmin=0, vmax=1, interpolation="nearest")
    ax[0, 1].set_title(f"Missing ({miss.sum()} = {miss.sum()/max(1, gm.sum()):.0%} of valid)")
    ax[1, 0].imshow(obs.astype(int), cmap=ListedColormap(["white", "royalblue"]), vmin=0, vmax=1, interpolation="nearest")
    ax[1, 0].set_title(f"Observed ({obs.sum()})")
    ax[1, 1].imshow(comp, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax[1, 1].set_title("Composite: outside / observed / missing")
    for axis in ax.ravel():
        axis.set_xticks(range(gm.shape[1]))
        axis.set_yticks(range(gm.shape[0]))
        axis.set_aspect("equal")
    fig.tight_layout()
    return fig, ax



def plot_training_history(history, title="Training curves"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    return fig, ax



def plot_error_histogram(errors, bins=80, density=True, title="Distribution of individual errors", xlabel="Error", ylabel="PDF"):
    import matplotlib.pyplot as plt

    errors = np.asarray(errors)
    fig, ax = plt.subplots()
    ax.hist(errors, bins=bins, density=density)
    ax.axvline(0.0, linestyle="--", color="k", label="0")
    if errors.size:
        ax.axvline(errors.mean(), linestyle="--", color="r", label="mean")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax



def plot_heatmap(values, title, colorbar_label, cmap=CUSTOM_32_CMAP, vmin=None, vmax=None, aspect="equal"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("j (col)")
    ax.set_ylabel("i (row)")
    return fig, ax



def plot_per_level_maps(maps, titles=None, cmap=CUSTOM_32_CMAP, vmin=None, vmax=None, colorbar_label="", figsize=(10, 9), separate_colorbars=False):
    import matplotlib.pyplot as plt

    maps = np.asarray(maps)
    num_levels = maps.shape[0]
    nrows, ncols = _per_level_grid(num_levels)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    flat_axes = np.atleast_1d(axs).ravel()
    level_titles = _resolve_level_titles(num_levels, titles)

    used_axes = []
    for i in range(num_levels):
        ax = flat_axes[i]
        if separate_colorbars:
            local_vmin = np.nanmin(maps[i]) if vmin is None and np.isfinite(maps[i]).any() else vmin
            local_vmax = np.nanmax(maps[i]) if vmax is None and np.isfinite(maps[i]).any() else vmax
        else:
            local_vmin = vmin
            local_vmax = vmax
        im = ax.imshow(maps[i], cmap=cmap, vmin=local_vmin, vmax=local_vmax, aspect="equal")
        ax.set_title(level_titles[i])
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        used_axes.append(ax)
        if separate_colorbars:
            cb = fig.colorbar(im, ax=ax, shrink=0.8)
            cb.set_label(colorbar_label)

    for ax in flat_axes[num_levels:]:
        ax.set_visible(False)

    if not separate_colorbars:
        fig.colorbar(im, ax=used_axes, shrink=0.75, label=colorbar_label)
    return fig, axs


def plot_per_level_error_histograms(errors_per_level, titles=None, bins=80, density=True, figsize=(10, 9)):
    import matplotlib.pyplot as plt

    errors_per_level = list(errors_per_level)
    num_levels = len(errors_per_level)
    nrows, ncols = _per_level_grid(num_levels)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    flat_axes = np.atleast_1d(axs).ravel()
    level_titles = _resolve_level_titles(num_levels, titles)

    for i in range(num_levels):
        ax = flat_axes[i]
        errors = np.asarray(errors_per_level[i])
        ax.hist(errors, bins=bins, density=density)
        ax.axvline(0.0, linestyle="--", color="k", label="0")
        if errors.size:
            ax.axvline(errors.mean(), linestyle="--", color="r", label="mean")
        ax.set_title(level_titles[i])
        ax.set_xlabel("Error")
        ax.set_ylabel("PDF" if density else "Count")
        ax.legend()

    for ax in flat_axes[num_levels:]:
        ax.set_visible(False)

    return fig, axs
