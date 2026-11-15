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
    figure.subplots_adjust(left=left, right=0.96, top=0.80, bottom=0.13)
    characters_per_line = int(size[0] * (0.96 - left) * 72 / 5.2)  # about 5.2 points per character at 9 points
    subtitle = textwrap.fill(subtitle, characters_per_line)
    axes.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 10), textcoords="offset points",
                  fontsize=9, color=SECONDARY, va="bottom", linespacing=1.4)
    lines = subtitle.count("\n") + 1
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
