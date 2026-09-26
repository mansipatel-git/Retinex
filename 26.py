import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from skimage.metrics import structural_similarity as ssim


# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = "Inp"

OUTPUT_FOLDER = "Retinex_Results"

# Number of images for final comparison
NUM_COMPARISON_IMAGES = 10


# ============================================================
# RETINEX PARAMETERS
# ============================================================

# MSR scales
SCALES = [15, 80, 250]

# SSR scale
SSR_SIGMA = 80

# MSRCR parameters
ALPHA = 125
BETA = 46

# Gain and offset
GAIN = 5.0
OFFSET = -25.0


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

OUTPUT_DIR = Path(OUTPUT_FOLDER)

SSR_DIR = OUTPUT_DIR / "SSR"
MSR_DIR = OUTPUT_DIR / "MSR"
MSRCR_DIR = OUTPUT_DIR / "MSRCR"

COMPARISON_DIR = OUTPUT_DIR / "Comparisons"
GRAPH_DIR = OUTPUT_DIR / "Graphs"

for folder in [
    OUTPUT_DIR,
    SSR_DIR,
    MSR_DIR,
    MSRCR_DIR,
    COMPARISON_DIR,
    GRAPH_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 1. SINGLE SCALE RETINEX (SSR)
# ============================================================

def ssr(channel, sigma):

    channel = channel.astype(np.float32)

    # Avoid log(0)
    channel = np.maximum(
        channel,
        1.0
    )

    # Estimate illumination
    blur = cv2.GaussianBlur(
        channel,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    blur = np.maximum(
        blur,
        1.0
    )

    # Retinex
    result = (
        np.log(channel)
        -
        np.log(blur)
    )

    return result


# ============================================================
# 2. MULTI SCALE RETINEX (MSR)
# ============================================================

def msr(
    image,
    scales=SCALES
):

    image = image.astype(
        np.float32
    )

    result = np.zeros_like(
        image,
        dtype=np.float32
    )

    # Process R, G, B
    for c in range(3):

        channel = image[:, :, c]

        channel_result = np.zeros_like(
            channel,
            dtype=np.float32
        )

        # SSR at multiple scales
        for sigma in scales:

            channel_result += ssr(
                channel,
                sigma
            )

        # Average
        channel_result /= len(
            scales
        )

        result[:, :, c] = channel_result

    return result


# ============================================================
# 3. COLOR RESTORATION
# ============================================================

def color_restoration(
    image,
    alpha=ALPHA,
    beta=BETA
):

    image = image.astype(
        np.float32
    )

    image = np.maximum(
        image,
        1.0
    )

    # R + G + B
    channel_sum = np.sum(
        image,
        axis=2,
        keepdims=True
    )

    channel_sum = np.maximum(
        channel_sum,
        1.0
    )

    color = beta * (
        np.log(alpha * image)
        -
        np.log(channel_sum)
    )

    return color


# ============================================================
# 4. MSRCR
# ============================================================

def msrcr(
    image,
    scales=SCALES,
    alpha=ALPHA,
    beta=BETA,
    gain=GAIN,
    offset=OFFSET
):

    # Step 1: MSR
    msr_result = msr(
        image,
        scales=scales
    )

    # Step 2: Color restoration
    color = color_restoration(
        image,
        alpha=alpha,
        beta=beta
    )

    # Step 3: Combine
    result = (
        color *
        msr_result
    )

    # Step 4: Gain + Offset
    result = gain * (
        result + offset
    )

    return result


# ============================================================
# 5. PERCENTILE NORMALIZATION
# ============================================================

def normalize_percentile(
    image,
    low_percentile=1,
    high_percentile=99
):

    image = image.astype(
        np.float32
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    # Process each channel
    for c in range(3):

        channel = image[:, :, c]

        low = np.percentile(
            channel,
            low_percentile
        )

        high = np.percentile(
            channel,
            high_percentile
        )

        if high - low < 1e-6:

            output[:, :, c] = 0

            continue

        # Clip extreme values
        channel = np.clip(
            channel,
            low,
            high
        )

        # Map to 0-255
        channel = (
            (channel - low)
            /
            (high - low)
            *
            255.0
        )

        output[:, :, c] = channel

    return np.clip(
        output,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# 6. MSE
# ============================================================

def calculate_mse(
    original,
    enhanced
):

    original = original.astype(
        np.float64
    )

    enhanced = enhanced.astype(
        np.float64
    )

    mse = np.mean(
        (original - enhanced) ** 2
    )

    return mse


# ============================================================
# 7. PSNR
# ============================================================

def calculate_psnr(
    original,
    enhanced
):

    mse = calculate_mse(
        original,
        enhanced
    )

    if mse < 1e-10:

        return float("inf")

    MAX_PIXEL = 255.0

    psnr_value = (
        10 *
        np.log10(
            (MAX_PIXEL ** 2)
            /
            mse
        )
    )

    return psnr_value


# ============================================================
# 8. SSIM
# ============================================================

def calculate_ssim(
    original,
    enhanced
):

    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2GRAY
    )

    enhanced_gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_RGB2GRAY
    )

    score = ssim(
        original_gray,
        enhanced_gray,
        data_range=255
    )

    return score


# ============================================================
# 9. UICM
# ============================================================

def calculate_uicm(image):

    image = image.astype(
        np.float64
    )

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # RG component
    RG = R - G

    # YB component
    YB = (
        (R + G) / 2.0
        -
        B
    )

    RG = RG.flatten()
    YB = YB.flatten()

    # Sort
    RG = np.sort(RG)
    YB = np.sort(YB)

    # Remove 10% extreme values
    trim = int(
        0.10 * len(RG)
    )

    if trim > 0:

        RG = RG[
            trim:-trim
        ]

        YB = YB[
            trim:-trim
        ]

    mean_RG = np.mean(RG)
    mean_YB = np.mean(YB)

    var_RG = np.mean(
        (RG - mean_RG) ** 2
    )

    var_YB = np.mean(
        (YB - mean_YB) ** 2
    )

    uicm = (
        -0.0268 *
        np.sqrt(
            mean_RG ** 2
            +
            mean_YB ** 2
        )
        +
        0.1586 *
        np.sqrt(
            var_RG
            +
            var_YB
        )
    )

    return uicm


# ============================================================
# 10. EME
# ============================================================

def calculate_eme(
    channel,
    block_size=8
):

    h, w = channel.shape

    num_x = int(
        np.ceil(
            h / block_size
        )
    )

    num_y = int(
        np.ceil(
            w / block_size
        )
    )

    total = 0.0

    for i in range(num_x):

        x1 = i * block_size

        x2 = min(
            (i + 1) * block_size,
            h
        )

        for j in range(num_y):

            y1 = j * block_size

            y2 = min(
                (j + 1) * block_size,
                w
            )

            block = channel[
                x1:x2,
                y1:y2
            ]

            block_min = np.min(
                block
            )

            block_max = np.max(
                block
            )

            if block_min <= 0:
                block_min = 1e-6

            if block_max <= 0:
                block_max = 1e-6

            if block_max > block_min:

                total += np.log(
                    block_max /
                    block_min
                )

    if num_x * num_y == 0:

        return 0.0

    return (
        2.0 /
        (num_x * num_y)
        *
        total
    )


# ============================================================
# 11. UISM
# ============================================================

def calculate_uism(image):

    image = image.astype(
        np.float64
    )

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Sobel edges
    R_edge = cv2.Sobel(
        R,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    G_edge = cv2.Sobel(
        G,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    B_edge = cv2.Sobel(
        B,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    # Absolute magnitude
    R_edge = np.abs(
        R_edge
    )

    G_edge = np.abs(
        G_edge
    )

    B_edge = np.abs(
        B_edge
    )

    # EME
    R_eme = calculate_eme(
        R_edge
    )

    G_eme = calculate_eme(
        G_edge
    )

    B_eme = calculate_eme(
        B_edge
    )

    # Weighted sharpness
    uism = (
        0.299 * R_eme
        +
        0.587 * G_eme
        +
        0.114 * B_eme
    )

    return uism


# ============================================================
# 12. UIConM
# ============================================================

def calculate_uiconm(image):

    image = image.astype(
        np.uint8
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    mean_value = np.mean(
        gray
    )

    contrast = np.mean(
        (gray - mean_value) ** 2
    )

    contrast = np.sqrt(
        contrast
    )

    return contrast


# ============================================================
# 13. UIQM
# ============================================================

def calculate_uiqm(image):

    # Standard coefficients
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    uicm = calculate_uicm(
        image
    )

    uism = calculate_uism(
        image
    )

    uiconm = calculate_uiconm(
        image
    )

    uiqm = (
        c1 * uicm
        +
        c2 * uism
        +
        c3 * uiconm
    )

    return (
        uiqm,
        uicm,
        uism,
        uiconm
    )


# ============================================================
# 14. FIND INPUT IMAGES
# ============================================================

input_path = Path(
    INPUT_FOLDER
)

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]

image_files = []

for extension in extensions:

    image_files.extend(
        input_path.glob(extension)
    )

# Remove duplicates
image_files = sorted(
    list(set(image_files))
)


# ============================================================
# CHECK IMAGES
# ============================================================

print("\n" + "=" * 80)

print(
    "       UNDERWATER RETINEX IMAGE ENHANCEMENT"
)

print("=" * 80)

print(
    f"\nInput folder: {INPUT_FOLDER}"
)

print(
    f"Images found: {len(image_files)}"
)

if len(image_files) == 0:

    print(
        "\nERROR: No images found in Inp folder!"
    )

    print(
        "\nPut your images inside:"
    )

    print(
        "Inp/"
    )

    raise SystemExit


# ============================================================
# SELECT FIRST 10 IMAGES
# ============================================================

comparison_files = image_files[
    :NUM_COMPARISON_IMAGES
]

print(
    f"\nImages selected: "
    f"{len(comparison_files)}"
)


# ============================================================
# RESULT LIST
# ============================================================

all_results = []


# ============================================================
# STORE IMAGES FOR 10 x 4 COMPARISON
# ============================================================

comparison_data = []


# ============================================================
# PROCESS EVERY IMAGE
# ============================================================

for image_number, image_path in enumerate(
    comparison_files,
    start=1
):

    print("\n" + "-" * 80)

    print(
        f"Image {image_number}/"
        f"{len(comparison_files)}"
    )

    print(
        f"Processing: {image_path.name}"
    )

    print("-" * 80)


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "ERROR loading image!"
        )

        continue


    # ========================================================
    # BGR → RGB
    # ========================================================

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # SSR
    # ========================================================

    print(
        "Running SSR..."
    )

    ssr_result = np.zeros_like(
        image_rgb,
        dtype=np.float32
    )

    for c in range(3):

        ssr_result[:, :, c] = ssr(
            image_rgb[:, :, c],
            SSR_SIGMA
        )

    ssr_display = normalize_percentile(
        ssr_result
    )


    # ========================================================
    # MSR
    # ========================================================

    print(
        "Running MSR..."
    )

    msr_result = msr(
        image_rgb,
        scales=SCALES
    )

    msr_display = normalize_percentile(
        msr_result
    )


    # ========================================================
    # MSRCR
    # ========================================================

    print(
        "Running MSRCR..."
    )

    msrcr_result = msrcr(
        image_rgb,
        scales=SCALES,
        alpha=ALPHA,
        beta=BETA,
        gain=GAIN,
        offset=OFFSET
    )

    msrcr_display = normalize_percentile(
        msrcr_result
    )


    # ========================================================
    # METHODS
    # ========================================================

    methods = {

        "SSR": ssr_display,

        "MSR": msr_display,

        "MSRCR": msrcr_display

    }


    # ========================================================
    # STORE IMAGES FOR FINAL 10 x 4 GRID
    # ========================================================

    comparison_data.append({

        "name": image_path.name,

        "Original": image_rgb,

        "SSR": ssr_display,

        "MSR": msr_display,

        "MSRCR": msrcr_display

    })


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    for method_name, enhanced in methods.items():

        print(
            f"Calculating "
            f"{method_name} metrics..."
        )


        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        mse_value = calculate_mse(
            image_rgb,
            enhanced
        )


        # ----------------------------------------------------
        # PSNR
        # ----------------------------------------------------

        psnr_value = calculate_psnr(
            image_rgb,
            enhanced
        )


        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------

        ssim_value = calculate_ssim(
            image_rgb,
            enhanced
        )


        # ----------------------------------------------------
        # UIQM
        # ----------------------------------------------------

        (
            uiqm_value,
            uicm_value,
            uism_value,
            uiconm_value
        ) = calculate_uiqm(
            enhanced
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        all_results.append({

            "Image": image_path.name,

            "Method": method_name,

            "MSE": mse_value,

            "PSNR": psnr_value,

            "SSIM": ssim_value,

            "UIQM": uiqm_value,

            "UICM": uicm_value,

            "UISM": uism_value,

            "UIConM": uiconm_value

        })


        # ====================================================
        # SAVE ENHANCED IMAGE
        # ====================================================

        if method_name == "SSR":

            save_folder = SSR_DIR

        elif method_name == "MSR":

            save_folder = MSR_DIR

        else:

            save_folder = MSRCR_DIR


        output_name = (
            image_path.stem
            +
            "_"
            +
            method_name
            +
            ".png"
        )

        output_path = (
            save_folder /
            output_name
        )


        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                enhanced,
                cv2.COLOR_RGB2BGR
            )
        )


# ============================================================
# 15. CREATE 10 x 4 COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("CREATING 10 x 4 COMPARISON")
print("=" * 80)


number_of_images = len(
    comparison_data
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    number_of_images,
    4,
    figsize=(12, 25),
    gridspec_kw={
        "wspace": 0,
        "hspace": 0
    }
)


# If only one image exists
if number_of_images == 1:

    axes = np.expand_dims(
        axes,
        axis=0
    )


# ============================================================
# COLUMN TITLES
# ============================================================

column_titles = [
    "Original",
    "SSR",
    "MSR",
    "MSRCR"
]


# ============================================================
# FILL GRID
# ============================================================

for row, data in enumerate(
    comparison_data
):

    images_to_show = [

        data["Original"],

        data["SSR"],

        data["MSR"],

        data["MSRCR"]

    ]


    for col in range(4):

        ax = axes[row, col]

        ax.imshow(
            images_to_show[col],
            aspect="auto"
        )

        # Remove everything around image
        ax.axis("off")

        # Column title only once
        if row == 0:

            ax.set_title(
                column_titles[col],
                fontsize=17,
                fontweight="bold",
                pad=3
            )


# ============================================================
# REMOVE ALL SPACING
# ============================================================

plt.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=0.965,
    wspace=0,
    hspace=0
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Underwater Image Enhancement - Retinex Comparison",
    fontsize=20,
    fontweight="bold",
    y=0.995
)


# ============================================================
# SAVE 10 x 4 FIGURE
# ============================================================

comparison_10x4_path = (
    COMPARISON_DIR /
    "All_10_Images_10x4_Comparison.png"
)

plt.savefig(
    comparison_10x4_path,
    dpi=200,
    bbox_inches="tight",
    pad_inches=0
)


print(
    "\n10 x 4 comparison saved:"
)

print(
    comparison_10x4_path
)

plt.show()

plt.close()


# ============================================================
# 16. CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# PRINT ALL RESULTS
# ============================================================

print("\n" + "=" * 100)

print(
    "                 RESULTS FOR ALL IMAGES"
)

print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE ALL RESULTS CSV
# ============================================================

all_results_csv = (
    OUTPUT_DIR /
    "Retinex_All_Results.csv"
)

results_df.to_csv(
    all_results_csv,
    index=False
)


# ============================================================
# 17. AVERAGE RESULTS
# ============================================================

average_df = (
    results_df
    .groupby("Method")
    [
        [
            "MSE",
            "PSNR",
            "SSIM",
            "UIQM",
            "UICM",
            "UISM",
            "UIConM"
        ]
    ]
    .mean()
    .reset_index()
)


# ============================================================
# PRINT AVERAGE
# ============================================================

print("\n" + "=" * 100)

print(
    "              AVERAGE RESULTS"
)

print("=" * 100)

print(
    average_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE AVERAGE CSV
# ============================================================

average_csv = (
    OUTPUT_DIR /
    "Retinex_Average_Results.csv"
)

average_df.to_csv(
    average_csv,
    index=False
)


# ============================================================
# 18. SAVE EXCEL
# ============================================================

excel_path = (
    OUTPUT_DIR /
    "Retinex_Results.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="All Images",
        index=False
    )

    average_df.to_excel(
        writer,
        sheet_name="Average",
        index=False
    )


# ============================================================
# 19. AVERAGE METRIC GRAPHS
# ============================================================

metrics = [
    "MSE",
    "PSNR",
    "SSIM",
    "UIQM"
]


for metric in metrics:

    plt.figure(
        figsize=(8, 6)
    )

    values = []

    for method in [
        "SSR",
        "MSR",
        "MSRCR"
    ]:

        value = average_df.loc[
            average_df["Method"] == method,
            metric
        ].values[0]

        values.append(
            value
        )


    plt.bar(
        [
            "SSR",
            "MSR",
            "MSRCR"
        ],
        values
    )


    plt.xlabel(
        "Retinex Method"
    )

    plt.ylabel(
        metric
    )

    plt.title(
        f"Average {metric}"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()


    graph_path = (
        GRAPH_DIR /
        f"Average_{metric}.png"
    )

    plt.savefig(
        graph_path,
        dpi=200
    )

    plt.show()

    plt.close()


# ============================================================
# 20. ALL AVERAGE METRICS IN ONE GRAPH
# ============================================================

normalized_df = average_df.copy()


for metric in metrics:

    min_value = normalized_df[
        metric
    ].min()

    max_value = normalized_df[
        metric
    ].max()

    if max_value - min_value > 1e-10:

        normalized_df[
            metric
        ] = (
            normalized_df[metric]
            -
            min_value
        ) / (
            max_value
            -
            min_value
        )

    else:

        normalized_df[
            metric
        ] = 0


x = np.arange(
    len(metrics)
)

width = 0.25


plt.figure(
    figsize=(12, 7)
)


for i, method in enumerate(
    [
        "SSR",
        "MSR",
        "MSRCR"
    ]
):

    values = normalized_df.loc[
        normalized_df["Method"] == method,
        metrics
    ].values[0]


    plt.bar(
        x + i * width,
        values,
        width,
        label=method
    )


plt.xticks(
    x + width,
    metrics
)

plt.xlabel(
    "Evaluation Metric"
)

plt.ylabel(
    "Normalized Average Value"
)

plt.title(
    "Average Performance Across 10 Images"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


combined_graph = (
    GRAPH_DIR /
    "Average_All_Metrics.png"
)


plt.savefig(
    combined_graph,
    dpi=200
)

plt.show()

plt.close()


# ============================================================
# 21. UIQM FOR EVERY IMAGE
# ============================================================

uiqm_table = results_df.pivot(
    index="Image",
    columns="Method",
    values="UIQM"
)


uiqm_table.plot(
    kind="bar",
    figsize=(16, 7)
)


plt.xlabel(
    "Input Image"
)

plt.ylabel(
    "UIQM"
)

plt.title(
    "UIQM Comparison for All Images"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


uiqm_graph = (
    GRAPH_DIR /
    "UIQM_All_Images.png"
)


plt.savefig(
    uiqm_graph,
    dpi=200
)

plt.show()

plt.close()


# ============================================================
# 22. SSIM FOR EVERY IMAGE
# ============================================================

ssim_table = results_df.pivot(
    index="Image",
    columns="Method",
    values="SSIM"
)


ssim_table.plot(
    kind="bar",
    figsize=(16, 7)
)


plt.xlabel(
    "Input Image"
)

plt.ylabel(
    "SSIM"
)

plt.title(
    "SSIM Comparison for All Images"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


ssim_graph = (
    GRAPH_DIR /
    "SSIM_All_Images.png"
)


plt.savefig(
    ssim_graph,
    dpi=200
)

plt.show()

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)

print(
    "              PROCESSING COMPLETED"
)

print("=" * 80)

print(
    f"\nTotal images processed: "
    f"{len(comparison_data)}"
)

print(
    "\nInput folder:"
)

print(
    f"    {INPUT_FOLDER}/"
)

print(
    "\nOutput folder:"
)

print(
    f"    {OUTPUT_FOLDER}/"
)

print("\nGenerated:")

print(
    "✓ SSR enhanced images"
)

print(
    "✓ MSR enhanced images"
)

print(
    "✓ MSRCR enhanced images"
)

print(
    "✓ 10 x 4 comparison image"
)

print(
    "✓ CSV results"
)

print(
    "✓ Average CSV"
)

print(
    "✓ Excel results"
)

print(
    "✓ MSE graph"
)

print(
    "✓ PSNR graph"
)

print(
    "✓ SSIM graph"
)

print(
    "✓ UIQM graph"
)

print(
    "✓ Combined metrics graph"
)

print(
    "✓ UIQM all-image graph"
)

print(
    "✓ SSIM all-image graph"
)

print("\nDone!")