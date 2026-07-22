#!/usr/bin/env python3
"""
nd2 -> single SVG export (parallel, scale bar, wavelength channel names)

Features:
- Uses ProcessPoolExecutor with up to min(20, os.cpu_count()) workers to process ND2 files in parallel.
- Groups positions from the same ND2 file together with a small spacing, and inserts a larger spacing between different ND2 files.
- Channels are named by reported wavelength when available (e.g., "405" or "405 nm"); falls back to "C0" etc.
- Adds a scale bar on the MERGE image using pixel size read from ND2 metadata.
- Embeds high-quality PNGs into a single SVG (keeps image fidelity).
- Colors can be set manually.
"""
import subprocess
import sys
import importlib
# ==========================================
# USER SETTINGS
# Set this to True if you are running the code for the first time
# Set this to False once your libraries are installed to save time
FIRST_TIME_SETUP = False


# ==========================================

def check_setup():
    if not FIRST_TIME_SETUP:
        return  # Skip the check entirely

    requirements = {
        "nd2": "nd2",
        "numpy": "numpy",
        "scipy": "scipy",
        "Pillow": "PIL",
        "matplotlib": "matplotlib"
    }

    print("--- Verifying environment... ---")
    for install_name, import_name in requirements.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"--- Library '{install_name}' missing. Installing... ---")
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
    print("--- Environment Ready! ---")


# Run the auto-installer check
check_setup()
#=============================================================

import os
import io
import base64
import math
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict, Any, List, Optional
from scipy.ndimage import binary_dilation, binary_erosion
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nd2 import ND2File


# === SETTINGS ===



# === Folder settings ===
INPUT_FOLDER = r"\\fs-tu.dzne.de\ag-schoels\Projects\TDP-43 Oligos\Data\Confocal\K-4\MBP"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "Export_ND2_to_SVG")
MASK_SUBFOLDER_NAME = "Masks"
#==================================================================
# === Single File Mode ===
# Set this to None to process ALL files in the folder.
# Set this to "your_file.nd2" to test only one specific file.
TEST_FILE = None


# === ORGANOID/SLICE MODE SETTINGS ===
# Set to True to enable mask-based centering and cropping.
ORGANOID_SLICE_MODE = False
MASK_CHANNEL_NAME = "405" # The channel to use for generating the mask (e.g., DAPI/DNA)
MASK_THRESHOLD_MODE = 'ABSOLUTE'
MASK_THRESHOLD_VALUE = 300 # The raw intensity value (if mode='ABSOLUTE') or percentile (if mode='PERCENTILE')
# Margin to add around the detected organoid boundary
# This defines the distance (in pixels) between the organoid's edge and the final crop box's edge.
MARGIN_PX = 200
# Set to 0 to skip, or a small odd integer (e.g., 3, 5, 7) for kernel size.
MORPH_KERNEL_SIZE = 7
#==================================================================

# === 96-well plate layout settings ===
USE_PLATE_LAYOUT = False
POINTS_PER_WELL = 5 # Defines how many points from the multipoint set per well are plotted
WELL_PATTERN = re.compile(r'(?:Well)?([A-H])(1[0-2]|0?[1-9])', re.IGNORECASE)
ROW_TO_INDEX = {row: i for i, row in enumerate('ABCDEFGH')}
# SVG constants for the plate layout
WELL_LABEL_WIDTH = 50
WELL_PADDING = 20


# === LUT auto-scaling settings ====
# Set to True to use fixed absolute min/max limits defined below (for specific channels).
# Set to False to use the AUTO_SCALE_PERCENTILES (default behavior).
USE_ABSOLUTE_SCALING = True
AUTO_SCALE_PERCENTILES = (5, 99.8)  # lower, upper percentiles for auto-scaling, optimal: 0.1 and 99.9; if too much noise higher to e.g. 3 and 99.9

# Per-channel Absolute Limits (Used if USE_ABSOLUTE_SCALING is True)
# Set manually {channel_name: (min_val, max_val)}.
# Any channel not listed will fall back to using AUTO_SCALE_PERCENTILES.

CHANNEL_ABSOLUTE_LIMITS: Dict[str, Optional[Tuple[float, float]]] = {
    "405": None,
    "488": None,
    "561": None,
    "638": None,
}
#==================================================================


# === SVG Settings ===

# Internal render quality (how detailed the PNGs are)
RENDER_WIDTH = 1024

# Layout settings svg file (pixels)
DISPLAY_WIDTH = 200          # width/height of each embedded PNG in SVG (square)
SPACING_X = 12
POSITION_SPACING_Y = 12      # spacing between positions of the same ND2
GROUP_SPACING_Y = 70         # extra spacing between different ND2 files
HEADER_HEIGHT = 64
TITLE_HEIGHT = 28
LEFT_COL_WIDTH = 220         # space for labels on left

# === Per-channel display settings ===
# Set True / False to control which channels are displayed in the SVG output.
# Even if False, they are still included in the merge.
CHANNEL_DISPLAY = {
    "405": True,
    "488": False,
    "561": True,
    "638": False,
    "MERGE": True, # allow turning off merged view if desired
}

# === Per-channel scale-bar visibility settings ===
# Set True / False to control which channels show a scale bar
SCALEBAR_VISIBILITY = {
    "405": False,
    "488": False,
    "561": False,
    "638": False,
    "MERGE": True,   # show on merged image too
}

# number of workers (use all cores up to 20)
MAX_WORKERS = min(20, (os.cpu_count() or 1))

# fallback colors if a channel has no manual mapping (cycled), you dont need it as long as every channel has a color assigned
FALLBACK_COLORS = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 0)
]


# Set colors manually, use HEX-codes or named matplotlib colors: https://matplotlib.org/3.4.2/gallery/color/named_colors.html
CHANNEL_COLORS_INPUT = {
    "405": "#0000FF", #blue: #0000FF
    "488": "green", #green: #00FF00
    "561": "yellow", #yellow: #FFF000
    "638": "yellow", #whitesmoke
}

# Commonly used (named) colors, copy and paste above to use
# blue
# smokewhite
# darkorchid
# red
# lime
# cyan


# === Optional custom channel name mapping ===
# Here you can name your channels so the images in the SVG are labelled with your specific marker/treatment.
# If disabled channel names are set to reflect the excitation wavelengths
USE_DIFFERENT_CHANNEL_NAMES = True   # Set False to disable replacements

CHANNEL_NAME_MAP = {
    "405": "DNA",
    "488": "Nogo-A",
    "561": "MBP",
    "638": "MBP",
}

# === Per-channel merge inclusion settings ===
# Set True / False to control whether each channel is included in the merged image.
CHANNEL_MERGE_INCLUDE = {
    "405": True,
    "488": False,
    "561": True,
    "638": False,
}

def _hex_or_name_to_rgb_tuple(s):
    """Convert CSS color names or hex strings to normalized RGB tuple (0..1)."""
    return mcolors.to_rgb(s)

CHANNEL_COLORS = {k: _hex_or_name_to_rgb_tuple(v) for k, v in CHANNEL_COLORS_INPUT.items()}


# === HELPERS ===
def _normalize_rgb(color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    arr = np.array(color, dtype=float)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return tuple(np.clip(arr, 0.0, 1.0).tolist())


def auto_scale_channel(img: np.ndarray,
                       lower_pct: float = 0.1,
                       upper_pct: float = 99.9,
                       abs_limits: Optional[Tuple[float, float]] = None
                       ) -> np.ndarray:
    if img.size == 0:
        return np.zeros_like(img, dtype=np.float32)

    # 1. Determine the limits (lo, hi)
    if abs_limits is not None:
        # Use provided absolute min/max values
         lo, hi = float(abs_limits[0]), float(abs_limits[1])
    else:
        # Fallback to auto-scaling by percentiles
        lo, hi = np.percentile(img, [lower_pct, upper_pct])

    if hi <= lo:
        maxv = float(img.max() if img.size else 1.0)
        if maxv <= 0:
            return np.zeros_like(img, dtype=np.float32)
        #Scale by max value
        return (img.astype(np.float32) / maxv).clip(0.0, 1.0)
    #Apply scaling by lo and hi
    scaled = (img.astype(np.float32) - float(lo)) / float(hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def colorize_channel(gray: np.ndarray, rgb: Tuple[float, float, float]) -> np.ndarray:
    rgb = _normalize_rgb(rgb)
    h, w = gray.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        rgb_img[..., i] = gray * rgb[i]
    return rgb_img


def float_rgb_to_png_bytes(img: np.ndarray, png_size: Tuple[int, int] = None) -> bytes:
    assert img.dtype in (np.float32, np.float64, np.uint8, np.uint16)
    if img.dtype != np.uint8:
        arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        arr = img
    pil = Image.fromarray(arr)
    if png_size is not None:
        pil = pil.resize(png_size, Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def img_bytes_to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("ascii")


# === ND2 METADATA UTILITIES ===
def _try_extract_pixel_size_um(f):
    """
    Extract pixel size (µm per pixel) from ND2 metadata.
    Tested for Nikon SoRa / confocal ND2 files with volume.axesCalibration fields.
    """
    # Try new ND2 SDK 1.3+ object structure
    try:
        ch0 = f.metadata.channels[0]
        if hasattr(ch0, "volume") and hasattr(ch0.volume, "axesCalibration"):
            cal = ch0.volume.axesCalibration
            if isinstance(cal, (tuple, list)) and len(cal) >= 2:
                px = float(cal[0])
                if px > 0:
                    return px
    except Exception:
        pass

    # Try frame metadata as backup
    try:
        fm = f.frame_metadata(0)
        ch0 = fm.channels[0]
        if hasattr(ch0, "volume") and hasattr(ch0.volume, "axesCalibration"):
            cal = ch0.volume.axesCalibration
            if isinstance(cal, (tuple, list)) and len(cal) >= 2:
                px = float(cal[0])
                if px > 0:
                    return px
    except Exception:
        pass

    # Fallbacks (old search patterns)
    for attr in ("pixel_size", "pixel_size_um", "pixel_microns", "pixel_size_x", "pixel_size_y"):
        val = getattr(f, attr, None)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return None


def _nice_round_length_um(length_um: float) -> float:
    """Round to a nice number like 1,2,5 * 10^n"""
    if length_um <= 0:
        return 0.0
    exponent = math.floor(math.log10(length_um))
    base = length_um / (10 ** exponent)
    if base < 1.5:
        nice = 1.0
    elif base < 3.5:
        nice = 2.0
    elif base < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10 ** exponent)

def draw_scalebar_if_enabled(img: np.ndarray, pixel_size_um: float, channel_name: str) -> np.ndarray:
    """Draw a scale bar on the image if SCALEBAR_VISIBILITY[channel_name] is True."""

    # If scalebar drawing is disabled or missing pixel size
    if not SCALEBAR_VISIBILITY.get(channel_name, False) or pixel_size_um is None or pixel_size_um <= 0:
        return img, None

    if pixel_size_um is None or pixel_size_um <= 0:
        return img

    h, w = img.shape[:2]
    img_pil = Image.fromarray((img * 255).astype("uint8"))
    draw = ImageDraw.Draw(img_pil)

    # Compute bar size (~1/3 of image width)
    fov_um = w * pixel_size_um
    bar_target = fov_um / 3.0 # change length of scalebar by higher/lower number; use 4.0 for shorter one
    bar_len_um = _nice_round_length_um(bar_target)
    bar_len_px = max(1, int(round(bar_len_um / pixel_size_um)))

    margin = int(round(0.05 * w))
    bar_height = max(6, int(round(0.01 * h)))

    # position on bottom-right
    bar_x1 = w - margin
    bar_x0 = bar_x1 - bar_len_px
    bar_y1 = h - margin - 20
    bar_y0 = bar_y1 - bar_height

    # Draw the bar only (not the text)
    draw.rectangle([bar_x0 - 1, bar_y0 - 1, bar_x1 + 1, bar_y1 + 1], fill=(0, 0, 0))
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(255, 255, 255))

    # Return normalized geometry (fractions) so SVG placement is resolution-independent
    scalebar_info = {
        "x0_frac": float(bar_x0) / float(w),
        "y0_frac": float(bar_y0) / float(h),
        "width_frac": float(bar_len_px) / float(w),
        "height_frac": float(bar_height) / float(h),
        "length_um": float(bar_len_um),
    }

    return np.asarray(img_pil).astype(np.float32) / 255.0, scalebar_info


def get_crop_bounds(mask_img: np.ndarray, margin_px: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Computes a square crop box that encompasses the mask's bounding box plus a margin.
    Applies morphological closing to fill small gaps in the mask before bounding.
    Returns (x_start, y_start, x_end, y_end)
    """
    if mask_img.size == 0:
        return None

    H, W = mask_img.shape

    # 1. Calculate Threshold Value based on Mode (reads global settings)
    global MASK_THRESHOLD_MODE, MASK_THRESHOLD_VALUE, MORPH_KERNEL_SIZE

    if MASK_THRESHOLD_MODE == 'ABSOLUTE':
        thresh_val = MASK_THRESHOLD_VALUE
    elif MASK_THRESHOLD_MODE == 'PERCENTILE':
        if MASK_THRESHOLD_VALUE >= 100:
            thresh_val = mask_img.max() + 1
        else:
            thresh_val = np.percentile(mask_img, MASK_THRESHOLD_VALUE)
    else:
        print(f"Error: Invalid MASK_THRESHOLD_MODE: {MASK_THRESHOLD_MODE}. Using default percentile 10.")
        thresh_val = np.percentile(mask_img, 10)

    # 2. Create Initial Binary Mask
    binary_mask = mask_img > thresh_val

    # --- MORPHOLOGICAL PROCESSING TO FILL HOLES (Closing) ---
    if MORPH_KERNEL_SIZE > 0:
        # Create a square kernel for the operation
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=bool)
        # Perform Closing: Dilation to fill gaps, followed by Erosion to restore size/shape
        dilated_mask = binary_dilation(binary_mask, structure=kernel)
        processed_mask = binary_erosion(dilated_mask, structure=kernel)
        binary_mask = processed_mask
    # --- END MORPHOLOGICAL PROCESSING ---

    if not np.any(binary_mask):
        print("Warning: Binary mask is empty. Cannot center crop.")
        return None

    # 3. Find the Bounding Box of the Masked Area
    coords = np.argwhere(binary_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 4. Apply Margin and Get Expanded Bounds
    exp_x_min = max(0, x_min - margin_px)
    exp_y_min = max(0, y_min - margin_px)

    exp_x_max = min(W, x_max + margin_px + 1)
    exp_y_max = min(H, y_max + margin_px + 1)

    # 5. Determine the Side Length for the Final Square Crop
    current_width = exp_x_max - exp_x_min
    current_height = exp_y_max - exp_y_min
    square_side = max(current_width, current_height)
    square_side = min(square_side, W, H)

    # 6. Center the Square Crop on the Expanded Box's Center
    center_x = (exp_x_min + exp_x_max) // 2
    center_y = (exp_y_min + exp_y_max) // 2
    half_side = square_side // 2

    # Recalculate final bounds for the square crop
    final_x_start = center_x - half_side
    final_x_end = final_x_start + square_side
    if final_x_end > W:
        final_x_end = W
        final_x_start = W - square_side

    final_y_start = center_y - half_side
    final_y_end = final_y_start + square_side
    if final_y_end > H:
        final_y_end = H
        final_y_start = H - square_side

    final_x_start = max(0, final_x_start)
    final_y_start = max(0, final_y_start)

    if (final_x_end - final_x_start) <= 0 or (final_y_end - final_y_start) <= 0:
        return None

    print(
        f"Crop: ({final_x_start}, {final_y_start}) to ({final_x_end}, {final_y_end}) - Size: {final_x_end - final_x_start}x{final_y_end - final_y_start}")

    return (final_x_start, final_y_start, final_x_end, final_y_end)

def apply_crop(pos_stack: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Applies crop bounds (x_start, y_start, x_end, y_end) to a (C, H, W) stack.
    """
    x_start, y_start, x_end, y_end = bounds
    # pos_stack is (C, Y, X)
    return pos_stack[:, y_start:y_end, x_start:x_end]

# ======= 96 well section =========
# === Helper Function: Extract Well Position ===
def _extract_well_position(filepath: str) -> Optional[str]:
    """
    Extracts the well position (e.g., 'B02', 'H12') from the filename,
    and returns a zero-padded column ID (e.g., 'B02' or 'A12').
    """
    filename = os.path.basename(filepath)

    # *** FIX 1: Use the pattern to search the filename ***
    match = WELL_PATTERN.search(filename)  # Use .search() or .match() method

    if match:
        # Group 1 is the Row letter (e.g., 'B')
        row = match.group(1).upper()

        # Group 2 is the Column number (e.g., '1', '02', '10')
        # Use zfill(2) to ensure the column is always two digits ('1' becomes '01')
        col = match.group(2).zfill(2)

        return f"{row}{col}"  # e.g., "B02" or "A10"

    return None


# === Helper Function: Extract Point ID ===
def _extract_point_id(filepath: str) -> Optional[str]:
    """
    Extracts the Point ID (e.g., 'P01') from the filepath.
    It expects a pattern like 'P1', 'P01', or 'P10'.
    Returns a zero-padded Point ID (e.g., 'P01').
    """
    filename = os.path.basename(filepath)
    # Tries to find P followed by one or more digits (case-insensitive)
    match = re.search(r'[Pp](\d+)', filename)

    if match:
        # Group 1 is the number (e.g., '1', '01', '10')
        point_num_str = match.group(1).zfill(2)
        return f"P{point_num_str}"  # Returns "P01", "P02", etc.

    return None


def _well_id_to_coordinates(well_id: str) -> Optional[Tuple[int, int]]:
    """
    Converts a zero-padded well ID (e.g., 'A01', 'B12') to matrix coordinates (row_idx, col_idx).
    A01 -> (0, 0), H12 -> (7, 11).
    """
    if not well_id or len(well_id) < 2:
        return None

    row_label = well_id[0].upper()
    col_label = well_id[1:]  # This is the zero-padded column string, e.g., '02' or '12'

    try:
        row_idx = ROW_TO_INDEX.get(row_label)
        # Convert column label ('01'...'12') to 0-based index (0...11)
        col_idx = int(col_label) - 1

        if row_idx is not None and 0 <= col_idx <= 11:
            return (row_idx, col_idx)
    except ValueError:
        # Fails if col_label is not a valid number
        return None

    return None


# Note: Reuse global constants like WELL_ROWS, WELL_COLS (for headers only), DISPLAY_WIDTH, etc.

def generate_plate_svg_matrix(plate_grid: List[List[Optional[Dict[str, Any]]]], out_path: str, points_per_well: int):
    """
    Generates an SVG for a 96-well plate layout based on a fixed 8x12 coordinate grid.
    """
    WELL_ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    WELL_COLS_HEADERS = [str(i) for i in range(1, 13)]  # Use non-padded numbers for display headers

    # 1. Determine the channel titles (must pull from the first non-None result)
    all_titles = set()
    found_data = False
    for row in plate_grid:
        for res in row:
            # FIX: Check for the 'data' key, which now holds the channel dictionary.
            if res and res.get('data'):
                # res['data'] is the dictionary of {'Channel_Name': (png_bytes, scalebar_info)}
                all_titles.update(t for t in res['data'].keys() if CHANNEL_DISPLAY.get(t, True))
                found_data = True
                break
        if found_data: break

    display_titles = sorted(list(all_titles))
    if not display_titles:
        raise ValueError("No displayable channels found in results.")

    img_per_well_row = len(display_titles) * points_per_well
    well_block_width = img_per_well_row * DISPLAY_WIDTH

    # Calculate dimensions
    svg_width = WELL_PADDING + WELL_LABEL_WIDTH + (len(WELL_COLS_HEADERS) * well_block_width) + WELL_PADDING
    svg_height = HEADER_HEIGHT + WELL_PADDING + (len(WELL_ROWS) * DISPLAY_WIDTH) + WELL_PADDING

    # --- 2. Start SVG Content ---
    parts = []
    # ... (SVG start, style, title code - reuse existing) ...
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">')
    parts.append(f'<style>text{{font-family: Arial, Helvetica, sans-serif}}</style>')
    parts.append(
        f'<text x="{svg_width / 2}" y="{30}" text-anchor="middle" font-size="20" font-weight="bold">96-Well Plate Layout (P={points_per_well})</text>')

    current_y = HEADER_HEIGHT + WELL_PADDING + 10
    X_START_GRID = WELL_PADDING + WELL_LABEL_WIDTH

    # --- Draw Column Headers (1-12) ---
    for col_idx, col_label in enumerate(WELL_COLS_HEADERS):
        x_start_well = X_START_GRID + (col_idx * well_block_width)
        x_center = x_start_well + (well_block_width / 2.0)
        parts.append(
            f'<text x="{x_center}" y="{current_y - 10}" text-anchor="middle" font-size="16" font-weight="bold">{col_label}</text>'
        )

    current_y += 10  # Move down after column labels

    # --- NEW: Draw Channel/Position Headers (e.g., "405", "MERGE", "405", "MERGE") ---
    # Determine the starting Y for the image row
    IMAGE_ROW_START_Y = current_y + 20  # Add space for channel labels

    # Loop through all 12 columns
    for col_idx in range(len(WELL_COLS_HEADERS)):
        x_start_well = X_START_GRID + (col_idx * well_block_width)
        current_x = x_start_well  # Reset X for the start of the well block

        # Loop through all points (P-axis)
        for p_idx in range(points_per_well):
            # Loop through all displayable channels
            for title_idx, title in enumerate(display_titles):
                x_center = current_x + DISPLAY_WIDTH / 2.0

                # Draw the channel title (e.g., "405", "MERGE")
                parts.append(
                    f'<text x="{x_center}" y="{current_y + 15}" text-anchor="middle" '
                    f'font-size="10" fill="#666666">{title}</text>'
                )

                current_x += DISPLAY_WIDTH  # Move X cursor to the next image slot

    current_y = IMAGE_ROW_START_Y

    # --- Draw the Plate Grid by iterating through the fixed matrix ---
    for row_idx, row_label in enumerate(WELL_ROWS):

        # Draw Row Header (A-H)
        x_row_label = WELL_PADDING + WELL_LABEL_WIDTH / 2.0
        y_center = current_y + DISPLAY_WIDTH / 2.0
        parts.append(
            f'<text x="{x_row_label}" y="{y_center}" text-anchor="middle" alignment-baseline="middle" font-size="16" font-weight="bold">{row_label}</text>'
        )

        for col_idx, res in enumerate(plate_grid[row_idx]):  # Iterate over the data in the row

            # Calculate the starting X-coordinate based *only* on the col_idx
            x_start_of_well_block = X_START_GRID + (col_idx * well_block_width)
            current_x = x_start_of_well_block

            if res is None or not res.get('data'):
                # Draw empty block for missing data
                parts.append(
                    f'<rect x="{current_x}" y="{current_y}" width="{well_block_width}" height="{DISPLAY_WIDTH}" '
                    f'fill="#eeeeee" stroke="#cccccc" stroke-width="1"/>'
                )
            else:
                # --- Well Data Found: Draw Images ---
                # FIX: Channel dictionary is now directly available under the 'data' key
                chdict = res['data']

                # Loop through all displayable channels (This handles a single position/point)
                for title in display_titles:
                    # The X cursor is already at the correct starting position (current_x) for the channel
                    if title in chdict:
                        img_bytes, scalebar_info = chdict[title]  # Get data and scalebar info
                        b64 = img_bytes_to_b64(img_bytes)

                        # Draw the main image
                        parts.append(
                            f'<image x="{current_x}" y="{current_y}" width="{DISPLAY_WIDTH}" height="{DISPLAY_WIDTH}" '
                            f'href="data:image/png;base64,{b64}" />'
                        )

                        # Draw scalebar label if info available (re-using the logic from the non-plate SVG generator)
                        if scalebar_info is not None:
                            sb = scalebar_info

                            # Convert normalized coords to SVG/display coords.
                            svg_bar_x = current_x + sb["x0_frac"] * DISPLAY_WIDTH
                            svg_bar_w = sb["width_frac"] * DISPLAY_WIDTH
                            svg_bar_y = current_y + sb["y0_frac"] * DISPLAY_WIDTH
                            # Max ensures height is at least 1 pixel
                            svg_bar_h = max(1, sb["height_frac"] * DISPLAY_WIDTH)

                            text_x = svg_bar_x + svg_bar_w / 2.0
                            text_y = svg_bar_y - 8

                            parts.append(
                                f'<text x="{text_x:.1f}" y="{text_y:.1f}" font-size="14" font-weight="bold" '
                                f'fill="white" stroke="black" stroke-width="0.5" text-anchor="middle">'
                                f'{int(sb["length_um"])} µm</text>'
                            )

                        # --- Channel label box (top-right corner) ---
                        box_w = DISPLAY_WIDTH / 2.5
                        box_h = 24
                        box_x = current_x + DISPLAY_WIDTH - box_w
                        box_y = current_y

                        # Determine label text color
                        if title == "MERGE":
                            txt_color = "black"
                        else:
                            rgb = CHANNEL_COLORS.get(title, (0, 0, 0))
                            txt_color = f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"

                        # --- Determine which channel label to display ---
                        if USE_DIFFERENT_CHANNEL_NAMES:
                            display_name = CHANNEL_NAME_MAP.get(title, title)
                        else:
                            display_name = title

                        # Draw semi-transparent white background and channel name text
                        parts.append(
                            f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" '
                            f'fill="white" fill-opacity="0.6" stroke="none" />'
                        )
                        parts.append(
                            f'<text x="{box_x + box_w / 2}" y="{box_y + box_h * 0.7}" '
                            f'font-size="18" font-weight="bold" text-anchor="middle" '
                            f'fill="{txt_color}" stroke="black" stroke-width="0.4" '
                            f'stroke-linejoin="round">{display_name}</text>'
                        )

                    current_x += DISPLAY_WIDTH  # Move to the next channel slot in the well
        current_y += DISPLAY_WIDTH

        # ... (SVG finalization and writing to file) ...
    parts.append("</svg>")
    svg_text = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(svg_text)
    return out_path




# === PROCESSING SINGLE ND2 (worker) ===
def process_nd2_file_worker(filepath: str):
    """
    Worker-friendly wrapper that reads ND2 file, performs Max Intensity Projection (MIP)
    on the correct Z-axis using f.sizes, and returns a structured result.
    """
    well_id = _extract_well_position(filepath)  # Add this

    try:
        with ND2File(filepath) as f:
            arr = f.asarray()
            if arr.ndim < 2:
                raise ValueError("ND2 array has too few dimensions")

            arr_shape = arr.shape
            height, width = arr_shape[-2], arr_shape[-1]
            leading = arr_shape[:-2]

            # Use f.sizes for robust dimension counting
            n_positions = f.sizes.get('P', 1)
            n_channels = f.sizes.get('C', 1)
            # Dimensions to check and project
            dims_to_project = {}
            if f.sizes.get('Z', 1) > 1:
                dims_to_project['Z'] = f.sizes['Z']
            if f.sizes.get('T', 1) > 1:
                dims_to_project['T'] = f.sizes['T']

            print(
                f"File: {os.path.basename(filepath)} has shape: {arr_shape}. Dimensions to project: {dims_to_project}")

            # --- 1. SEQUENTIAL PROJECTION (Z and/or T) ---

            # We must project sequentially, as array indices shift after each projection
            for dim_name, dim_size in dims_to_project.items():

                # Find the axis index corresponding to this dimension size
                axis_index = -1
                for i, size in enumerate(leading):
                    if size == dim_size:
                        axis_index = i
                        break

                if axis_index != -1:
                    print(f"Performing MaxIP on {dim_name}-axis (index {axis_index}) with size {dim_size}.")

                    # Perform Max Intensity Projection (MIP)
                    arr = arr.max(axis=axis_index)

                    # Update the array's "leading" dimensions for the next iteration
                    leading_list = list(leading)
                    leading_list.pop(axis_index)
                    leading = tuple(leading_list)

                else:
                    print(
                        f"⚠️ {dim_name}-axis (size {dim_size}) not found in remaining leading dimensions {leading}. Skipping projection.")

            # --- 2. FINAL RESHAPING TO (P, C, H, W) ---

            if len(leading) == 0:
                n_positions, n_channels = 1, 1
            elif len(leading) == 1:
                # After MIP, this should be (C, H, W) or (P, H, W)
                n_channels = leading[0] if n_channels > 1 else 1
                n_positions = leading[0] if n_channels == 1 else 1
            elif len(leading) >= 2:
                # After MIP, this should be (P, C, H, W)
                n_positions = leading[0]
                n_channels = leading[1]

            # Final reshape to the rendering format (P, C, H, W)
            arr = arr.reshape((n_positions, n_channels, height, width))

            # === Extract channel wavelengths / names from ND2 metadata ===
            # (Channel name logic remains the same, as it relies on n_channels)
            channel_wavelengths: List[Any] = [None] * n_channels
            channel_names: List[str] = []

            try:
                if hasattr(f, "metadata") and hasattr(f.metadata, "channels"):
                    for ch in f.metadata.channels:
                        meta = ch.channel
                        wl = getattr(meta, "excitationLambdaNm", None)
                        name = getattr(meta, "name", None)

                        if wl is not None:
                            name_clean = str(int(round(wl)))
                        elif name:
                            name_clean = name.split()[0]
                        else:
                            name_clean = f"C{len(channel_names)}"

                        channel_names.append(name_clean)
                else:
                    channel_names = [f"C{c}" for c in range(n_channels)]

            except Exception as e:
                print(f"⚠️ Could not extract channel names from metadata: {e}")
                channel_names = [f"C{c}" for c in range(n_channels)]

            # attempt to get pixel size (um)
            pixel_size_um = _try_extract_pixel_size_um(f)

            filename = os.path.splitext(os.path.basename(filepath))[0]
            result = {}
            for pos in range(arr.shape[0]):
                pos_stack = arr[pos]
                crop_bounds = None
                if ORGANOID_SLICE_MODE:
                    try:
                        # 1. Find the index of the mask channel
                        mask_ch_idx = channel_names.index(MASK_CHANNEL_NAME)
                        mask_img_raw = pos_stack[mask_ch_idx]  # (H, W)

                        # 2. Calculate crop bounds
                        crop_bounds = get_crop_bounds(
                            mask_img_raw,
                            MARGIN_PX,
                        )
                        if MASK_SUBFOLDER_NAME:
                            # 1. Create mask saving path and directory
                            mask_save_dir = os.path.join(OUTPUT_FOLDER, MASK_SUBFOLDER_NAME)
                            os.makedirs(mask_save_dir, exist_ok=True)

                            # 2. Re-generate binary mask using the same logic as get_crop_bounds
                            if MASK_THRESHOLD_MODE == 'ABSOLUTE':
                                thresh_val = MASK_THRESHOLD_VALUE
                            elif MASK_THRESHOLD_MODE == 'PERCENTILE':
                                if MASK_THRESHOLD_VALUE >= 100:
                                    thresh_val = mask_img_raw.max() + 1
                                else:
                                    thresh_val = np.percentile(mask_img_raw, MASK_THRESHOLD_VALUE)
                            else:
                                print(
                                    f"Error: Invalid MASK_THRESHOLD_MODE: {MASK_THRESHOLD_MODE} during mask save. Skipping.")
                                continue  # skip saving this mask if mode is invalid

                                # Generate binary mask (True/False array)
                            binary_mask = mask_img_raw > thresh_val
                            # --- APPLY MORPHOLOGY FOR SAVING ---
                            global MORPH_KERNEL_SIZE  # Needed to access the global setting
                            if MORPH_KERNEL_SIZE > 0:
                                kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=bool)
                                # Perform Closing: Dilation followed by Erosion
                                dilated_mask = binary_dilation(binary_mask, structure=kernel)
                                processed_mask = binary_erosion(dilated_mask, structure=kernel)
                                binary_mask = processed_mask

                            # 3. Convert to 8-bit image (0=background, 255=mask) for saving as PNG
                            mask_8bit = (binary_mask.astype(np.uint8) * 255)
                            mask_pil = Image.fromarray(mask_8bit, mode='L')  # 'L' is 8-bit grayscale

                            # 4. Save the file
                            mask_filename = f"{filename}_pos{pos}_Mask.png"
                            mask_path = os.path.join(mask_save_dir, mask_filename)

                            mask_pil.save(mask_path)
                            print(f"Saved binary mask: {mask_filename}")

                        # 3. Apply crop to the whole stack if successful
                        if crop_bounds:
                            print(f"Applying margin-based square crop to {filename} position {pos}...")
                            # pos_stack is updated here with the cropped array
                            pos_stack = apply_crop(pos_stack, crop_bounds)
                        else:
                            print(f"Could not generate mask for {filename} position {pos}. Skipping crop.")

                    except ValueError:
                        print(f"⚠️ MASK_CHANNEL_NAME '{MASK_CHANNEL_NAME}' not found. Skipping crop for pos {pos}.")
                    except Exception as e:
                        print(f"⚠️ Error during mask/crop for pos {pos}: {e}. Skipping crop.")


                h, w = pos_stack.shape[1], pos_stack.shape[2]
                merged = np.zeros((h, w, 3), dtype=np.float32)
                per_channel_imgs: Dict[str, np.ndarray] = {}

                # (The rest of the channel processing and rendering loops remain unchanged)
                for ch_idx in range(pos_stack.shape[0]):
                    ch_img_raw = pos_stack[ch_idx]
                    ch_wl = channel_names[ch_idx]
                    scaling_limits = None
                    if USE_ABSOLUTE_SCALING:
                        scaling_limits = CHANNEL_ABSOLUTE_LIMITS.get(ch_wl)
                    ch_scaled = auto_scale_channel(ch_img_raw,
                                                   AUTO_SCALE_PERCENTILES[0],
                                                   AUTO_SCALE_PERCENTILES[1],
                                                   abs_limits=scaling_limits
                                                   )

                    # ... (Color selection logic as before) ...
                    keys_to_try = []
                    if ch_wl is not None:
                        clean = str(ch_wl).replace("nm", "").replace("(SingleEM)", "").strip()
                        clean = clean.split()[0]
                        if clean.endswith(".0"): clean = clean[:-2]
                        keys_to_try += [clean, str(int(float(clean))) if clean.replace('.', '').isdigit() else clean]
                    keys_to_try.append(ch_idx)

                    rgb = None
                    for k in keys_to_try:
                        if k in CHANNEL_COLORS:
                            rgb = CHANNEL_COLORS[k]
                            break
                    if rgb is None:
                        try:
                            meta_color = getattr(f.metadata.channels[ch_idx].channel, "color", None)
                            if meta_color is not None:
                                rgb = (meta_color.r / 255.0, meta_color.g / 255.0, meta_color.b / 255.0)
                        except Exception:
                            pass
                    if rgb is None:
                        rgb = FALLBACK_COLORS[ch_idx % len(FALLBACK_COLORS)]
                    # ... (end color selection) ...

                    rgb = _normalize_rgb(rgb)
                    rgb_img = colorize_channel(ch_scaled, rgb)
                    rgb_img_with_bar, scalebar_info = draw_scalebar_if_enabled(rgb_img, pixel_size_um, ch_wl)
                    per_channel_imgs[ch_wl] = (rgb_img_with_bar, scalebar_info)

                    # Only include in merge if allowed
                    if CHANNEL_MERGE_INCLUDE.get(ch_wl, True):
                        merged += rgb_img

                merged = np.clip(merged, 0.0, 1.0)

                merged_with_bar, merge_bar_info = draw_scalebar_if_enabled(merged, pixel_size_um, "MERGE")
                per_channel_imgs["MERGE"] = (merged_with_bar, merge_bar_info)

                # Convert to PNG bytes (resized to DISPLAY_WIDTH square)
                png_size = (RENDER_WIDTH, int(RENDER_WIDTH * h / w))
                repname = f"{filename}_pos{pos}"
                rep_dict = {}
                for title, img_data in per_channel_imgs.items():
                    if isinstance(img_data, tuple):
                        img, scalebar_info = img_data
                    else:
                        img, scalebar_info = img_data, None
                    png = float_rgb_to_png_bytes(img, png_size=png_size)
                    rep_dict[title] = (png, scalebar_info)

                result[repname] = rep_dict

            return {
                "file": filename,
                "filepath": filepath,
                "pixel_size_um": pixel_size_um,
                "results": result,
                "well_position": well_id
            }
    except Exception as e:
        print(f"Error reading ND2 file {filepath}: {e}")
        return {"file": os.path.splitext(os.path.basename(filepath))[0], "filepath": filepath, "pixel_size_um": None,
                "results": {}}


# === SVG GENERATION ===
def generate_svg(grouped_ordered: List[Dict[str, Any]], out_path: str):
    """
    grouped_ordered: list of dicts returned by process_nd2_file_worker, keeping order.
    Writes a single SVG file embedding PNGs.
    """
    # Build rows: each row is (file_label, repname, titles, chdict)
    rows = []
    max_cols = 0
    # Store rows with their folder path for layout calculation
    svg_rows_with_path = []

    for entry in grouped_ordered:
        # Extract the directory name (the subfolder name)
        file_dir = os.path.dirname(entry["filepath"])
        folder_label = os.path.basename(file_dir) or "Root"
        filelabel = entry["file"]
        for repname, chdict in entry["results"].items():
            titles = sorted(chdict.keys())
            max_cols = max(max_cols, len(titles))
            svg_rows_with_path.append((folder_label, filelabel, repname, titles, chdict))

    if not svg_rows_with_path:
        raise ValueError("No images to write to SVG")


    # compute SVG dimensions
    y = HEADER_HEIGHT + 10
    heights = []
    total_height = HEADER_HEIGHT
    last_folder = None
    # We'll need to know when the file label changes to insert extra spacing
    svg_rows = []
    last_file = None
    for entry in grouped_ordered:
        filelabel = entry["file"]
        # get positions for this file
        for repname, chdict in entry["results"].items():
            svg_rows.append((filelabel, repname, sorted(chdict.keys()), chdict))
    # compute height summing with group/position spacing

    for folder_label, filelabel, repname, titles, chdict in svg_rows_with_path:
        # Add space for the Folder Header
        if last_folder is None or folder_label != last_folder:
            # We need space for the folder header and a larger group spacing
            if last_folder is not None:
                y += GROUP_SPACING_Y  # Spacing between the end of the last group and new header
            y += TITLE_HEIGHT + 15  # Space for the new folder title/header

        # Add spacing between positions within the same file/group
        elif filelabel != svg_rows_with_path[svg_rows_with_path.index((folder_label, filelabel, repname, titles, chdict)) - 1][1]:
            y += POSITION_SPACING_Y
        else:
            y += POSITION_SPACING_Y

        heights.append((folder_label, filelabel, repname, titles, chdict, y))
        y += DISPLAY_WIDTH
        last_folder = folder_label
    svg_height = max(y + 20, HEADER_HEIGHT + 300)

    svg_width = LEFT_COL_WIDTH + (DISPLAY_WIDTH * max_cols) + (SPACING_X * (max_cols + 1))

    # --- 2. Write SVG Content ---
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">')
    parts.append(f'<style>text{{font-family: Arial, Helvetica, sans-serif}}</style>')
    parts.append(f'<text x="{svg_width/2}" y="{30}" text-anchor="middle" font-size="20" font-weight="bold">ND2 export</text>')

    y_cursor = HEADER_HEIGHT + 10
    last_folder_label = None
    last_filelabel = None
    for folderlabel, filelabel, repname, titles, chdict, start_y in heights:
        # INSERT SUBFOLDER HEADER
        if last_folder_label is None or folderlabel != last_folder_label:

            # --- Draw Separator Line ---
            if last_folder_label is not None:
                parts.append(
                    f'<line x1="10" y1="{start_y - (TITLE_HEIGHT / 2) - 10}" x2="{svg_width - 10}" y2="{start_y - (TITLE_HEIGHT / 2) - 10}" stroke="#cccccc" stroke-width="1"/>')

            # --- Draw Folder Label/Header ---
            parts.append(
                f'<text x="12" y="{start_y - 8}" font-size="18" font-weight="bold" fill="#333333">{folderlabel}</text>')

            last_filelabel = None  # Reset file label tracking when folder changes

        if last_filelabel is None or filelabel != last_filelabel:
            # Draw File Label (ND2 file name) - remains on the left
            parts.append(f'<text x="12" y="{start_y + 14}" font-size="14" font-weight="bold">{filelabel}</text>')

        # Draw Position Label (Repname)
        parts.append(
             f'<text x="{12}" y="{start_y + DISPLAY_WIDTH / 2}" font-size="12" alignment-baseline="middle">{repname.split("_pos")[-1]}</text>')  # only show position number


        # place channel titles and images
        x = LEFT_COL_WIDTH
        for t in titles:
            # Skip if channel not marked for display
            if not CHANNEL_DISPLAY.get(t, True):
                continue
            img_bytes, scalebar_info = chdict[t]
            b64 = img_bytes_to_b64(img_bytes)

            # Draw the main image
            parts.append(
                f'<image x="{x}" y="{start_y}" width="{DISPLAY_WIDTH}" height="{DISPLAY_WIDTH}" href="data:image/png;base64,{b64}" />')

            # Draw scalebar label if info available (normalized geometry)
            if scalebar_info is not None:
                sb = scalebar_info

                # Convert normalized coords to SVG/display coords.
                # The embedded PNG is displayed as DISPLAY_WIDTH wide in the SVG,
                # so x positions are DISPLAY_WIDTH * fraction, anchored at x.
                svg_bar_x = x + sb["x0_frac"] * DISPLAY_WIDTH
                svg_bar_w = sb["width_frac"] * DISPLAY_WIDTH

                # y uses DISPLAY_WIDTH for width, but the image is embedded square in SVG at
                # height DISPLAY_WIDTH as well (your current SVG uses height="{DISPLAY_WIDTH}"),
                # so we map y fraction -> DISPLAY_WIDTH too (keeps consistent visual position).
                svg_bar_y = start_y + sb["y0_frac"] * DISPLAY_WIDTH
                svg_bar_h = max(1, sb["height_frac"] * DISPLAY_WIDTH)

                # Draw (optional) svg rect overlay if you want — but we only need label position.
                # Center label above the drawn bar:
                text_x = svg_bar_x + svg_bar_w / 2.0
                text_y = svg_bar_y - 8  # a few pixels above the bar (in SVG coords)

                parts.append(
                    f'<text x="{text_x:.1f}" y="{text_y:.1f}" font-size="14" font-weight="bold" '
                    f'fill="white" stroke="black" stroke-width="0.5" text-anchor="middle">'
                    f'{int(sb["length_um"])} µm</text>'
                )


            # --- Channel label box (top-right corner) ---
            box_w = DISPLAY_WIDTH / 2.5
            box_h = 24  # enough for one text line
            box_x = x + DISPLAY_WIDTH - box_w # position x
            box_y = start_y # position y

            # Determine label text color
            if t == "MERGE":
                txt_color = "black"
            else:
                rgb = CHANNEL_COLORS.get(t, (0, 0, 0))
                txt_color = f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"

            # --- Determine which channel label to display ---
            if USE_DIFFERENT_CHANNEL_NAMES:
                display_name = CHANNEL_NAME_MAP.get(t, t)
            else:
                display_name = t

            # Draw semi-transparent white background and channel name text
            parts.append(
                f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" '
                f'fill="white" fill-opacity="0.6" stroke="none" />'
            )
            parts.append(
                f'<text x="{box_x + box_w / 2}" y="{box_y + box_h * 0.7}" '
                f'font-size="18" font-weight="bold" text-anchor="middle" '
                f'fill="{txt_color}" stroke="black" stroke-width="0.4" '
                f'stroke-linejoin="round">{display_name}</text>'
            )

            x += DISPLAY_WIDTH + SPACING_X

        last_folder_label = folderlabel
        last_filelabel = filelabel


    parts.append("</svg>")
    svg_text = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(svg_text)
    return out_path


# === MAIN ===
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    nd2_files = []
    for root, _, files in os.walk(INPUT_FOLDER):
        # avoid recursing into output folder
        if os.path.commonpath([os.path.abspath(root), os.path.abspath(OUTPUT_FOLDER)]) == os.path.abspath(OUTPUT_FOLDER):
            continue
        for f in files:
            if f.lower().endswith(".nd2"):
                nd2_files.append(os.path.join(root, f))
    nd2_files.sort()
    # --- Filter for a single test file ---
    if TEST_FILE:
        # We look for any path in our list that ends with our test filename
        nd2_files = [path for path in nd2_files if os.path.basename(path) == TEST_FILE]

        if not nd2_files:
            print(f"⚠️ TEST MODE: Could not find '{TEST_FILE}' in {INPUT_FOLDER}")
            return
        else:
            print(f"🔬 TEST MODE: Processing only '{TEST_FILE}'")
    if not nd2_files:
        print(f"No ND2 files found in {INPUT_FOLDER}")
        return

    print(f"Processing {len(nd2_files)} ND2 files with up to {MAX_WORKERS} workers...")

    results_ordered = []
    # run in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # submit all
        futures = {ex.submit(process_nd2_file_worker, fp): fp for fp in nd2_files}
        for fut in as_completed(futures):
            fp = futures[fut]
            try:
                res = fut.result()
                print(f"Done: {fp} (pixel_size_um={res.get('pixel_size_um')})")
                results_ordered.append(res)
            except Exception as e:
                print(f"Worker failed for {fp}: {e}")
    # keep the same order as nd2_files by sorting results_ordered by filepath index
    path_to_idx = {p: i for i, p in enumerate(nd2_files)}
    results_ordered.sort(key=lambda r: path_to_idx.get(r.get("filepath", ""), 9999))

    if USE_PLATE_LAYOUT:
        print("Generating 96-well plate layout...")
        results_by_point = {}

        # 1. Grouping Logic
        for res in results_ordered:
            if res.get('results'):
                def natural_sort_key(s):
                    return [int(text) if text.isdigit() else text.lower()
                            for text in re.split('([0-9]+)', s)]

                sorted_point_keys = sorted(res['results'].keys(), key=natural_sort_key)

                for i, point_key in enumerate(sorted_point_keys):
                    if i >= POINTS_PER_WELL:
                        break

                    data = res['results'][point_key]
                    point_id = point_key.split('_')[-1] if '_' in point_key else f'pos{i}'

                    if point_id not in results_by_point:
                        results_by_point[point_id] = []

                    point_res = res.copy()
                    point_res['data'] = data
                    results_by_point[point_id].append(point_res)

        # 2. Generation Logic
        for point_id, point_list in results_by_point.items():
            print(f"Creating plate layout for point: {point_id}")
            plate_grid = [[None for _ in range(12)] for _ in range(8)]

            for res in point_list:
                coords = _well_id_to_coordinates(res.get("well_position"))
                if coords:
                    row_idx, col_idx = coords
                    plate_grid[row_idx][col_idx] = res
                else:
                    print(f"Warning: No well position for {res.get('filepath')}")

            svg_filename = f"96_well_plate_{point_id}.svg"
            svg_path = os.path.join(OUTPUT_FOLDER, svg_filename)
            generate_plate_svg_matrix(plate_grid, svg_path, 1)
            print(f"Saved: {svg_filename}")

    else:
        print("Generating standard linear layout...")
        svg_path = os.path.join(OUTPUT_FOLDER, "all_images_grid.svg")
        generate_svg(results_ordered, svg_path)
        print(f"Saved SVG to: {svg_path}")


if __name__ == "__main__":
    main()