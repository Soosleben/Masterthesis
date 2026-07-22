import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog, simpledialog


# Keep text editable in SVG

plt.rcParams["svg.fonttype"] = "none"

# File selection (Excel or CSV)

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

# Load file
if file_path.lower().endswith(".csv"):
    df = pd.read_csv(
        file_path,
        sep=None,  # auto-detect delimiter
        engine="python",  # needed for auto-detection
        encoding="utf-8"  # safe default
    )
else:
    df = pd.read_excel(file_path)

# Figure size input

width_cm = simpledialog.askfloat(
    "Figure Width",
    "Enter figure width (cm): (7 for two samples, 8.5 for three)",
    initialvalue=7
)

height_cm = simpledialog.askfloat(
    "Figure Height",
    "Enter figure height (cm):",
    initialvalue=7
)

# Color palette

palette = simpledialog.askstring(
    "Color Palette",
    "Enter colormap (e.g. Set2, Set3, tab10, viridis):",
    initialvalue="#d9d9d9"
)

# Keep required columns

df = df[[
    "FilenameOf405 (SingleEM)",
    "Dapi low intens total count",
    "MBP DAPI low intens count"
]]

# Convert to numeric (handles comma decimals)
for col in [
    "Dapi low intens total count",
    "MBP DAPI low intens count"
]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# QC filter: remove low DAPI counts

df = df[df["Dapi low intens total count"] >= 1]

# Remove invalid rows
df = df.dropna(subset=[
    "Dapi low intens total count",
    "MBP DAPI low intens count"
])


# Calculate MBP positive percentage

df["MBP pos cells %"] = (
    df["MBP DAPI low intens count"] /
    df["Dapi low intens total count"] *
    100
)
#Quality control
#df = df[
#    (df["MBP pos cells %"] > 0) &
 #   (df["MBP pos cells %"] < 100)
#]

# Group by filename

grouped = (
    df.groupby(
        "FilenameOf405 (SingleEM)", sort=False)["MBP pos cells %"]
    .apply(list)
)

filenames = grouped.index.tolist()
values = grouped.tolist()

# Figure setup

cm_to_inch = 1 / 2.54

fig, ax = plt.subplots(
    figsize=(
        width_cm * cm_to_inch,
        height_cm * cm_to_inch
    )
)
# Boxplot

bp = ax.boxplot(
    values,
    patch_artist=True,
    showfliers=False,
    widths=0.6
)

# Color boxes

try:
    cmap = plt.get_cmap(palette)
    colors = cmap(np.linspace(0.2, 0.8, len(values)))
except:
    colors = ["lightgray"] * len(values)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor("black")

# Scatter overlay (individual measurements)

rng = np.random.default_rng(42)

for i, y in enumerate(values, start=1):
    x = rng.normal(i, 0.06, len(y))

    ax.scatter(
        x,
        y,
        color="black",
        s=5,
        alpha=0.5,
        zorder=3
    )

# Labels
ax.set_ylabel(r"MBP$^{+}$ cells (%) over DAPI")
ax.set_xticks(range(1, len(filenames) + 1))
ax.set_xticklabels(filenames)
#ax.set_xticklabels(filenames, rotation=45, ha="right")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0,100)
plt.tight_layout()

# Save SVG

save_path = filedialog.asksaveasfilename(
    title="Save SVG",
    defaultextension=".svg",
    filetypes=[("SVG files", "*.svg")]
)

if save_path:
    fig.savefig(save_path, format="svg", bbox_inches="tight")

plt.show()