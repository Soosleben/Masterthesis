import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog, simpledialog

# Keep text editable in SVG
plt.rcParams["svg.fonttype"] = "none"

# ----------------------------
# File selection
# ----------------------------
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select Excel or CSV file",
    filetypes=[
        ("Data files", "*.xlsx *.xls *.csv"),
        ("Excel files", "*.xlsx *.xls"),
        ("CSV files", "*.csv")
    ]
)

if not file_path:
    raise SystemExit("No file selected.")

# ----------------------------
# Load data
# ----------------------------
if file_path.lower().endswith(".csv"):
    df = pd.read_csv(
        file_path,
        sep=None,
        engine="python",
        encoding="utf-8"
    )
else:
    df = pd.read_excel(file_path)

# ----------------------------
# Figure size
# ----------------------------
width_cm = simpledialog.askfloat(
    "Figure Width",
    "Enter figure width (cm):",
    initialvalue=7
)

height_cm = simpledialog.askfloat(
    "Figure Height",
    "Enter figure height (cm):",
    initialvalue=7
)

# ----------------------------
# Bar color
# ----------------------------
bar_color = simpledialog.askstring(
    "Bar Color",
    "Enter a color (e.g. lightgray, #d9d9d9, cornflowerblue):",
    initialvalue="#d9d9d9"
)

# ----------------------------
# Keep required columns
# ----------------------------
df = df[[
    "FilenameOf405 (SingleEM)",
    "Dapi low intens total count"
]]

# ----------------------------
# Convert to numeric
# ----------------------------
df["Dapi low intens total count"] = (
    df["Dapi low intens total count"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

df["Dapi low intens total count"] = pd.to_numeric(
    df["Dapi low intens total count"],
    errors="coerce"
)

# Remove invalid rows
df = df.dropna(subset=["Dapi low intens total count"])

# Optional QC
df = df[df["Dapi low intens total count"] >= 1]

# ----------------------------
# Calculate mean
# ----------------------------
means = (
    df.groupby("FilenameOf405 (SingleEM)", sort=False)
      ["Dapi low intens total count"]
      .mean()
)
filenames = means.index.tolist()
mean_values = means.values

# ----------------------------
# Figure
# ----------------------------
cm_to_inch = 1 / 2.54

fig, ax = plt.subplots(
    figsize=(
        width_cm * cm_to_inch,
        height_cm * cm_to_inch
    )
)

# ----------------------------
# Bar plot
# ----------------------------
x = np.arange(len(filenames))

ax.bar(
    x,
    mean_values,
    width=0.3,
    color=bar_color,
    edgecolor="black",
    linewidth=1.2,
    zorder=1
)

# ----------------------------
# Overlay all individual points
# ----------------------------
rng = np.random.default_rng(42)

for i, name in enumerate(filenames):

    y = df.loc[
        df["FilenameOf405 (SingleEM)"] == name,
        "Dapi low intens total count"
    ]

    jitter = rng.normal(0, 0.06, len(y))

    ax.scatter(
        np.full(len(y), i) + jitter,
        y,
        color="black",
        s=5,
        alpha=0.5,
        zorder=3
    )

# ----------------------------
# Labels
# ----------------------------
ax.set_ylabel("DAPI low intensity total count")
ax.set_xticks(x)
ax.set_xticklabels(filenames)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# ----------------------------
# Save SVG
# ----------------------------
save_path = filedialog.asksaveasfilename(
    title="Save SVG",
    defaultextension=".svg",
    filetypes=[("SVG files", "*.svg")]
)

if save_path:
    fig.savefig(save_path, format="svg", bbox_inches="tight")

plt.show()