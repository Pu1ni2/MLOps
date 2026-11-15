"""
The charts logged with each evaluation. They share one style: a palette checked for color-blind
readers, thin lines, light grids, and text in ink colors rather than in the series color.
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

BLUE = "#2a78d6"
LIGHT_BLUE = "#86b6ef"
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# One hue from light to dark, for counts: the more wines in a cell, the darker it is
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.labelcolor": MUTED, "axes.labelsize": 9,
})


def new_chart(title, subtitle, size=(6.4, 4.8), left=0.12):
    """
    An empty chart with a title and a subtitle above it, both aligned with the plot's left edge. The
    subtitle wraps to the plot's width; left is the room kept for the y-axis labels (share of the width).
    """
    figure, axes = plt.subplots(figsize=size)
    characters_per_line = int(size[0] * (0.96 - left) * 72 / 5.2)  # about 5.2 points per character at 9 points
    subtitle = textwrap.fill(subtitle, characters_per_line)
    lines = subtitle.count("\n") + 1
    # Room above the plot for the title and every line of the subtitle, in points
    room = 16 + 13 * lines + 26
    figure.subplots_adjust(left=left, right=0.96, top=1 - room / (size[1] * 72), bottom=0.13)
    axes.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 10), textcoords="offset points",
                  fontsize=9, color=SECONDARY, va="bottom", linespacing=1.4)
    axes.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 16 + 13 * lines), textcoords="offset points",
                  fontsize=12, fontweight="bold", color=INK, va="bottom")
    return figure, axes


def roc_chart(truth, chance, auc):
    """How many high-quality wines the model finds (y) for each share of ordinary wines it wrongly flags (x)."""
    false_positive_rate, true_positive_rate, _ = roc_curve(truth, chance)
    figure, axes = new_chart("ROC curve", f"AUC {auc:.3f} on {len(truth):,} wines. The diagonal is random guessing.")
    axes.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    axes.plot(false_positive_rate, true_positive_rate, color=BLUE, linewidth=2, solid_capstyle="round")
    axes.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Share of ordinary wines flagged as high quality",
             ylabel="Share of high-quality wines found")
    axes.grid(True)
    return figure


def precision_recall_chart(truth, chance, average_precision):
    """How precise the model's high-quality calls are (y) for each share of high-quality wines it finds (x)."""
    precision, recall, _ = precision_recall_curve(truth, chance)
    share = float(np.mean(truth))
    figure, axes = new_chart("Precision and recall", f"Average precision {average_precision:.3f}. The flat line is "
                                                     f"random guessing ({share:.0%} of these wines are high quality).")
    axes.plot([0, 1], [share, share], color=AXIS, linewidth=1)
    axes.plot(recall, precision, color=BLUE, linewidth=2, solid_capstyle="round")
    axes.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall: share of high-quality wines found",
             ylabel="Precision: high-quality calls that are right")
    axes.grid(True)
    return figure


def confusion_chart(truth, chance):
    """The four outcomes when a wine is called high quality at a chance of 50% or more."""
    counts = confusion_matrix(truth, chance >= 0.5)
    figure, axes = new_chart("Confusion matrix", "A wine is called high quality when the model gives it a "
                                                 "chance of 50% or more", size=(5.6, 4.8), left=0.24)
    axes.imshow(counts, cmap=BLUES)
    for (actual, predicted), count in np.ndenumerate(counts):
        on_dark = count > counts.max() * 0.45  # white text on the darker cells
        axes.text(predicted, actual, f"{count:,}", ha="center", va="center", fontsize=14,
                  color="#ffffff" if on_dark else INK)
    labels = ["ordinary", "high quality"]
    axes.set_xticks([0, 1], labels, color=SECONDARY, fontsize=10)
    axes.set_yticks([0, 1], labels, color=SECONDARY, fontsize=10)
    axes.set(xlabel="Predicted", ylabel="Actual")
    for side in ("left", "bottom"):
        axes.spines[side].set_visible(False)
    return figure


def tuning_chart(scores, reference=None):
    """Each attempt's validation AUC as a dot, the best so far as a line, and the best baseline as a grey line."""
    attempts = np.arange(1, len(scores) + 1)
    subtitle = f"Validation AUC of each of the {len(scores)} attempts (dots) and the best so far (line)."
    if reference is not None:
        subtitle += f" Grey line: the best baseline ({reference:.3f})."
    figure, axes = new_chart("Tuning progress", subtitle)
    if reference is not None:
        axes.axhline(reference, color=MUTED, linewidth=1)
    axes.scatter(attempts, scores, s=22, color=LIGHT_BLUE, edgecolors=SURFACE, linewidths=1, zorder=2)
    axes.step(attempts, np.maximum.accumulate(scores), where="post", color=BLUE, linewidth=2, zorder=3)
    axes.set(xlabel="Attempt", ylabel="Validation AUC", xlim=(0.5, len(scores) + 0.5))
    axes.grid(True, axis="y")
    return figure


def leakage_chart(rows):
    """
    One row per setup, with a dot for its AUC on each group of test wines: all of them, the ones it had
    already seen in training, and the new ones. rows: setup, auc_all, auc_seen, auc_new (seen may be missing).
    """
    groups = [("auc_all", "all test wines", BLUE), ("auc_seen", "already seen in training", "#eb6834"),
              ("auc_new", "new wines only", "#1baf7a")]
    # Without any seen wines, "all" and "new" are the same wines: show only the new ones
    rows = [row if row.get("auc_seen") is not None else {**row, "auc_all": None} for row in rows]
    figure, axes = new_chart("Duplicates inflated the original lab's score",
                             "Test AUC on all test wines, on the copies of training wines among them, and on the "
                             "new wines", size=(8.0, 4.6), left=0.40)
    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows):
        values = [row[key] for key, _, _ in groups if row.get(key) is not None]
        axes.plot([min(values), max(values)], [position, position], color=GRID, linewidth=2, zorder=1)
    for key, label, color in groups:
        points = [(row[key], position) for row, position in zip(rows, positions) if row.get(key) is not None]
        axes.scatter(*zip(*points), s=60, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=label)
        for value, position in points:
            axes.annotate(f"{value:.3f}", (value, position), xytext=(0, 9), textcoords="offset points",
                          ha="center", fontsize=8.5, color=SECONDARY)
    axes.set_yticks(positions, [row["setup"] for row in rows], color=SECONDARY, fontsize=9)
    axes.set(xlabel="Test AUC", ylim=(-0.6, len(rows) - 0.3))
    axes.grid(True, axis="x")
    axes.spines["left"].set_visible(False)
    axes.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=SECONDARY)
    return figure


def importance_chart(table):
    """Ranked bars: how much the AUC drops when each measurement's values are shuffled."""
    rows = table.sort_values("auc_drop")  # barh draws from the bottom, so the largest ends up on top
    figure, axes = new_chart("What the model relies on", "Drop in AUC when one measurement's values are shuffled. "
                                                         "The grey line is the spread over 5 shuffles.",
                             size=(6.4, 5.2), left=0.27)
    positions = np.arange(len(rows))
    axes.barh(positions, rows["auc_drop"], height=0.5, color=BLUE)
    axes.errorbar(rows["auc_drop"], positions, xerr=rows["spread"], fmt="none", ecolor=MUTED, elinewidth=1)
    axes.set_yticks(positions, rows["feature"], color=SECONDARY, fontsize=9)
    axes.set_xlabel("Drop in AUC")
    axes.grid(True, axis="x")
    axes.spines["left"].set_visible(False)
    return figure
