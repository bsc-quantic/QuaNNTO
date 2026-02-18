import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

from quannto.core.data_processors import softmax_discretization
from quannto.utils.path_utils import figures_dir

colors = matplotlib.cm.tab10(range(6))
linestyles = [
    (5, (10, 3)),
    (0, (3, 1, 1, 1)),
    (0, (5, 5)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (5, 1)),
    (0, (3, 5, 1, 5))]

def plot_noisy_dataset(task_name, train_dataset, real_function, 
                       figsize=(5, 4), fontsize=14, legend_fontsize=13):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_dataset[0], train_dataset[1], 'go', label='Noisy training set')
    ax.plot(real_function[0], real_function[1], 'b', label='Real function')
    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_ylabel('f(x)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(linestyle='--', linewidth=0.4)
    ax.legend(loc='best', fontsize=legend_fontsize)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"training_{task_name}.pdf", bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_qnns_testing(inputs, expected_outputs, qnns_outputs, legend_labels, filename,
                      title=None, figsize=(5.5, 4.2), fontsize=14, title_fontsize=14, legend_fontsize=13):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(inputs, expected_outputs, c='black', linewidth=7.0, alpha=0.25, label='Expected results')
    c = 0
    for (qnn_outputs, legend_label, linestyle) in zip(qnns_outputs, legend_labels, linestyles):
        ax.plot(inputs, np.real_if_close(qnn_outputs), c=colors[c], linestyle=linestyle, linewidth=1.8, label=legend_label)
        c += 1
    if title is not None:
        ax.set_title(f'{title}', fontsize=title_fontsize)

    ax.set_xlabel('Input', fontsize=fontsize)
    #ax.set_ylabel('Output', fontsize=fontsize)
    #plt.ylim(top=np.max(expected_outputs) + len(qnns_outputs)*0.5 + 0.3)
    ax.tick_params(axis='both', labelsize=fontsize)

    ax.grid(linestyle='--', linewidth=0.4)
    ax.legend(loc='best', fontsize=legend_fontsize)

    fig.tight_layout()
    fig.savefig(figures_dir() / f"test_{filename}.pdf", bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_qnns_loglosses(train_losses, valid_losses, legend_labels, filename,
                        figsize=(5, 4), fontsize=14, title_fontsize=14, legend_fontsize=13):
    fig, ax = plt.subplots(figsize=figsize)

    if valid_losses is not None and len(valid_losses[0]) > 1:
        ax.plot([], [], linestyle='dotted', label='Validation losses', c='black')
        title = 'TRAINING AND VALIDATION LOSS (log)'
    else:
        title = 'TRAINING LOSS (log)'

    log_train_losses = [np.log(np.array(loss) + 1) for loss in train_losses]
    if valid_losses is not None and len(valid_losses[0]) > 1:
        log_valid_losses = [np.log(np.array(loss) + 1) for loss in valid_losses]
    for i in range(len(legend_labels)):
        ax.plot(log_train_losses[i],
                c=colors[i], linestyle=linestyles[i],
                label=f'{legend_labels[i]}')

        if valid_losses is not None and len(valid_losses[i]) > 1:
            ax.plot(log_valid_losses[i],
                    c=colors[i], linestyle='dotted')

    double_max = 3 * np.max(np.array([np.min(loss) for loss in log_train_losses]))
    ax.set_ylim(bottom=0.0, top=double_max)
    ax.set_xlabel('Epochs', fontsize=fontsize)
    ax.set_ylabel('Loss value', fontsize=fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(linestyle='--', linewidth=0.4)
    ax.legend(fontsize=legend_fontsize)

    fig.tight_layout()
    fig.savefig(figures_dir() / f"loss_{filename}.pdf", bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print('=== ACHIEVED TRAINING LOSSES ===')
    for i in range(len(legend_labels)):
        print(f'{legend_labels[i]}: {train_losses[i][-1]}')
        
def plot_confusion_matrix(model_name, expected_cats, qnn_pred_cats):
    # Generate the confusion matrix
    cm = confusion_matrix(expected_cats, qnn_pred_cats)
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Plotting the confusion matrix as a green heatmap with variable opacity
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', alpha=cm_normalized)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(figures_dir() / f"cm_{model_name}.pdf", bbox_inches="tight")
    #plt.show()
    plt.clf()
    
def plot_qnn_decision(X, y, qonn_outputs, model_name, title="QONN decision boundary"):
    # 1) Scatter original data
    plt.figure(figsize=(6,6))
    plt.scatter(
        X[:,0], X[:,1],
        c=y, cmap="coolwarm",
        edgecolor="k", s=40, alpha=0.8
    )
    # 2) Create grid
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 3) Predict probabilities on grid
    probs = softmax_discretization(qonn_outputs(grid))[:,0] # probability of class=1
    Z = probs.reshape(xx.shape)
    # 4) Plot soft‐background and decision contour
    plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.3)
    plt.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)
    # 5) Set titles and limits
    plt.title(title, fontsize=18)
    plt.xticks(color='black', fontsize=14)
    plt.yticks(color='black', fontsize=14)
    plt.xlabel("$x_1$", color='black', fontsize=16)
    plt.ylabel("$x_2$", color='black', fontsize=16)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(figures_dir() / f"{model_name}.pdf", bbox_inches="tight")
    plt.show()
    plt.clf()
    
def plot_per_class_accuracy_hist(
    classes,
    y_true,
    preds_list,
    *,
    legend_labels=None,
    filename=None, 
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

    if legend_labels is None:
        legend_labels = [f"Model {i+1}" for i in range(n_models)]
    if len(legend_labels) != n_models:
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

    for i, name in enumerate(legend_labels):
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
    plt.savefig(figures_dir() / f"hist_{filename}.pdf", bbox_inches="tight")
    plt.show()
    plt.clf()
    return fig, ax, acc_matrix

def plot_per_class_accuracy_markers(
    classes,
    y_true,
    preds_list,
    avg_accuracies,
    *,
    legend_labels=None,
    filename=None,
    markers=None,              # e.g. ['^','o','s']
    linestyles=None,           # e.g. ['-','--',':','-.']
    cmap_name="tab10",
    figsize=(10, 5),
    title="Per-class accuracy (markers & lines)",
    annotate=False,
    x_jitter=0.0,
    legend_loc="upper center",
    legend_anchor=(0.5, 1.0), # slightly above the top edge
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

    if legend_labels is None:
        legend_labels = [f"Model {i+1}" for i in range(n_models)]
    if len(legend_labels) != n_models:
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

    for i, name in enumerate(legend_labels):
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

    ylim = (np.min(acc_matrix)-0.03, 1.07)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel("Classes")
    ax.set_ylabel("Per-class predictions accuracy")
    ax.set_yticks(np.arange(0, 1.01, 0.05))
    ax.set_ylim(ylim)
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
    plt.savefig(figures_dir() / f"acchist_{filename}.pdf", bbox_inches="tight")
    plt.show()
    plt.clf()
    print('=== ACCURACIES ACHIEVED ===')
    for i in range(len(legend_labels)):
        print(f'{legend_labels[i]}: {avg_accuracies[i]}')
    return fig, ax, acc_matrix