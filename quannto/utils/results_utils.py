import numpy as np
import matplotlib.pyplot as plt

def plot_qnn_testing(qnn, exp_outputs, qnn_outputs):
    plt.plot(exp_outputs, 'go', label='Expected results')
    plt.plot(qnn_outputs, 'r', label='QNN results')
    plt.title(f'TESTING SET\nModel: {qnn.model_name}, Modes = {qnn.N}, Layers = {qnn.layers}')
    plt.legend()
    plt.savefig(f"figures/testset_{qnn.model_name}_{qnn.N}modes_{qnn.layers}layers_{qnn.n_in}in.pdf")
    
def plot_qnn_train_results(qnn, inputs, exp_outputs, qnn_outputs, loss_values):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'TRAINING SET\nModel: {qnn.model_name}, Modes = {qnn.N}, Layers = {qnn.layers}')
    
    # Plot expected vs obtained outputs of the training set
    ax1.plot(inputs, exp_outputs,'go',label='Expected results')
    ax1.plot(inputs, qnn_outputs,'r',label='QNN results')
    ax1.set_xlabel('Input (x)')
    ax1.set_ylabel('Output')
    ax1.grid(linestyle='--', linewidth=0.4)
    ax1.legend()
    
    # Plot training loss values
    ax2.plot(np.log(np.array(loss_values)+1), 'r', label='Loss (logarithmic) function')
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Logarithmic loss value')
    ax2.grid(linestyle='--', linewidth=0.4)
    ax2.legend()
    plt.show()
    
def plot_per_class_accuracy_hist(
    classes,
    y_true,
    preds_list,
    *,
    model_names=None,
    cmap_name="tab10",
    figsize=(10, 5),
    title="Per-class accuracy",
    annotate=True,
    rotation=0,
    ylim=(0.0, 1.0)
):
    """
    Plot a grouped bar chart of per-class accuracy for multiple models.

    Parameters
    ----------
    classes : list or array-like
        Class labels (in the desired x-axis order). Can be ints or strings.
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    preds_list : list of array-like
        List where each element is y_pred for ONE model (same length as y_true).
        Example: [y_pred_model1, y_pred_model2, ...].
    model_names : list of str, optional
        Names for the models (same order as preds_list). Defaults to ["Model 1", ...].
    cmap_name : str
        Matplotlib colormap name to color the models (default: "tab10").
    figsize : tuple
        Figure size.
    title : str
        Figure title.
    annotate : bool
        If True, annotate each bar with its accuracy percentage.
    rotation : int or float
        X tick label rotation.
    ylim : tuple
        Y-axis limits (default 0..1).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    acc_matrix : np.ndarray of shape (n_models, n_classes)
        Per-class accuracy matrix.
    """
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)
    n_classes = len(classes)

    # Validate preds_list
    if not isinstance(preds_list, (list, tuple)) or len(preds_list) == 0:
        raise ValueError("preds_list must be a non-empty list of prediction arrays.")
    preds_np = []
    for k, preds in enumerate(preds_list):
        arr = np.asarray(preds)
        if arr.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Length mismatch for model {k}: y_pred ({arr.shape[0]}) vs y_true ({y_true.shape[0]})."
            )
        preds_np.append(arr)
    n_models = len(preds_np)

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]
    if len(model_names) != n_models:
        raise ValueError("model_names length must match preds_list length.")

    # Compute per-class accuracy
    acc_matrix = np.full((n_models, n_classes), np.nan, dtype=float)
    for j, cls in enumerate(classes):
        mask = (y_true == cls)
        denom = int(mask.sum())
        if denom == 0:
            continue  # keep NaN if class not present in y_true
        yt_cls = y_true[mask]
        for i, y_pred in enumerate(preds_np):
            acc_matrix[i, j] = (y_pred[mask] == yt_cls).mean()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.get_cmap(cmap_name)

    x = np.arange(n_classes, dtype=float)
    width = min(0.8 / max(n_models, 1), 0.8)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    for i, name in enumerate(model_names):
        heights = np.nan_to_num(acc_matrix[i], nan=0.0)
        color = cmap(i % cmap.N)  # cycle colors if >10 models
        bars = ax.bar(x + offsets[i], heights, width, label=name, color=color)
        if annotate:
            for bar, v in zip(bars, acc_matrix[i]):
                if np.isnan(v):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.01,
                    f"{v*100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes], rotation=rotation)
    ax.set_ylabel("Per-class accuracy")
    ax.set_title(title)
    ax.set_ylim(top=1.2)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    #ax.legend(title="Model", ncols=min(n_models, 5))
    ax.legend()
    fig.tight_layout()
    return fig, ax, acc_matrix

def plot_per_class_accuracy_markers(
    classes,
    y_true,
    preds_list,
    avg_accuracies,
    *,
    model_names=None,
    markers=None,              # e.g. ['^','o','s']
    linestyles=None,           # e.g. ['-','--',':','-.']
    cmap_name="tab10",
    figsize=(10, 5),
    title="Per-class accuracy (markers & lines)",
    annotate=False,
    x_jitter=0.0,
    ylim=(0.8, 1.0),           # keep (0.8, 1.0) as requested
    legend_loc="upper center", # legend stays INSIDE the axes
    legend_anchor=(0.5, 0.98), # slightly below the top edge
):
    """Plot per-class accuracy for multiple models using colored markers & lines."""
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)
    n_classes = len(classes)

    # validate inputs
    if not isinstance(preds_list, (list, tuple)) or len(preds_list) == 0:
        raise ValueError("preds_list must be a non-empty list of prediction arrays.")
    preds_np = []
    for k, preds in enumerate(preds_list):
        arr = np.asarray(preds)
        if arr.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Length mismatch for model {k}: y_pred ({arr.shape[0]}) vs y_true ({y_true.shape[0]})."
            )
        preds_np.append(arr)
    n_models = len(preds_np)

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]
    if len(model_names) != n_models:
        raise ValueError("model_names length must match preds_list length.")

    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':']

    # per-class accuracy (empirical probability of correct prediction)
    acc_matrix = np.full((n_models, n_classes), np.nan, dtype=float)
    for j, cls in enumerate(classes):
        mask = (y_true == cls)
        denom = int(mask.sum())
        if denom == 0:
            continue
        yt_cls = y_true[mask]
        for i, y_pred in enumerate(preds_np):
            acc_matrix[i, j] = (y_pred[mask] == yt_cls).mean()

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.get_cmap(cmap_name)
    x = np.arange(n_classes, dtype=float)
    offsets = np.zeros(1) if n_models == 1 else (np.arange(n_models) - (n_models - 1)/2.0) * x_jitter

    for i, name in enumerate(model_names):
        y_vals = acc_matrix[i]
        color = cmap(i % cmap.N)
        marker = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]

        ax.plot(
            x + offsets[i], y_vals,
            linestyle=ls, color=color, linewidth=2,
            marker=marker, markersize=7, label=name, alpha=1.0
        )
        # horizontal line at model's average accuracy (same color & linestyle)
        ax.axhline(
            avg_accuracies[i],
            color=color, linestyle=ls, linewidth=2.5, alpha=0.5, zorder=3
        )
        if annotate:
            for xx, yy in zip(x + offsets[i], y_vals):
                if np.isnan(yy): 
                    continue
                ax.text(xx, yy + 0.01, f"{yy*100:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel("Classes")
    ax.set_ylabel("Per-class predictions accuracy")
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(ylim[0], ylim[1]+0.01, 0.05))
    ax.set_title(title)

    # Legend INSIDE the axes, centered at the top (like before, but not outside)
    leg = ax.legend(
        #title="Model",
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,  # inside: y < 1.0
        ncols=min(n_models, 2),
        frameon=True
    )
    leg.get_frame().set_alpha(0.9)

    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    return fig, ax, acc_matrix

    
def compute_times(qnn):
    qnn_times = qnn.qnn_profiling.avg_benchmark()
    print('\nAverage time per stage per iteration:')
    print(qnn_times)
    print('\nAverage time usage per stage:')
    total_time = sum(list(qnn_times.values()))
    for part_time in qnn_times:
        print(f'\t {part_time}: {np.round(100 * qnn_times[part_time] / total_time, 3)} %')
    print(f'\nTotal average time per iteration: {total_time}')
    return qnn_times
    
def show_times(qnn):
    qnn_times = compute_times(qnn)
    plt.figure(figsize=(14,5))
    plt.bar(list(qnn_times.keys()), list(qnn_times.values()), color ='maroon')
    plt.xlabel("Time category")
    plt.ylabel("Time (s)")
    plt.title("QNN training times")
    plt.show()

    print(f'\nTotal number of training iterations: {len(qnn.qnn_profiling.gauss_times)}')
    #print(f'\tNumber of trace expressions: {len(qnn.ladder_modes)*len(qnn.ladder_modes[0])}')
    #print(f'\tNumber of perfect matchings per expression: {len(qnn.perf_matchings)}')
    #print(f'\t{len(qnn.perf_matchings)*len(qnn.ladder_modes)*len(qnn.ladder_modes[0])} total summations with {qnn.layers + 1} products per summation.')
