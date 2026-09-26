

#--------------
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_IMAGE = "hazy.png"

# SSR scale
SSR_SIGMA = 80

# MSR scales
MSR_SCALES = [15, 80, 250]

# Color restoration parameters
ALPHA = 125.0
BETA = 46.0

# Gain and offset
GAIN = 5.0
OFFSET = -25.0


# ============================================================
# 1. SSR
# ============================================================

def ssr(image, sigma=80):

    image = image.astype(np.float32)

    # Avoid log(0)
    image = np.maximum(image, 1.0)

    # Estimate illumination
    blur = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    blur = np.maximum(blur, 1.0)

    # Retinex:
    # R(x) = log(I(x)) - log(G(x) * I(x))
    result = (
        np.log(image)
        -
        np.log(blur)
    )

    return result


# ============================================================
# 2. MSR
# ============================================================

def msr(image, scales=[15, 80, 250]):

    image = image.astype(np.float32)

    result = np.zeros_like(
        image,
        dtype=np.float32
    )

    # Process RGB channels
    for c in range(3):

        channel = image[:, :, c]

        channel_result = np.zeros_like(
            channel,
            dtype=np.float32
        )

        # Apply SSR at multiple scales
        for sigma in scales:

            channel_result += ssr(
                channel,
                sigma
            )

        # Equal weighting
        channel_result /= len(scales)

        result[:, :, c] = channel_result

    return result


# ============================================================
# 3. COLOR RESTORATION
# ============================================================

def color_restoration(
    image,
    alpha=125,
    beta=46
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

    # Color restoration
    color = beta * (
        np.log(
            alpha * image
        )
        -
        np.log(
            channel_sum
        )
    )

    return color


# ============================================================
# 4. MSRCR
# ============================================================

def msrcr(
    msr_input,
    original_image,
    alpha=125,
    beta=46,
    gain=5.0,
    offset=-25.0
):

    # --------------------------------------------------------
    # Color restoration is calculated from ORIGINAL image
    # --------------------------------------------------------

    color = color_restoration(
        original_image,
        alpha,
        beta
    )

    # --------------------------------------------------------
    # MSR result × Color Restoration
    # --------------------------------------------------------

    result = (
        msr_input * color
    )

    # --------------------------------------------------------
    # Gain + Offset
    # --------------------------------------------------------

    result = gain * (
        result + offset
    )

    return result


# ============================================================
# 5. NORMALIZATION
# ============================================================

def normalize(image):

    image = image.astype(
        np.float32
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    for c in range(3):

        channel = image[:, :, c]

        # Remove extreme values
        low = np.percentile(
            channel,
            1
        )

        high = np.percentile(
            channel,
            99
        )

        if high - low < 1e-6:
            continue

        channel = np.clip(
            channel,
            low,
            high
        )

        channel = (
            (channel - low)
            /
            (high - low)
            *
            255
        )

        output[:, :, c] = channel

    return np.clip(
        output,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# 6. MSE
# ============================================================

def calculate_mse(
    original,
    enhanced
):

    original = original.astype(
        np.float32
    )

    enhanced = enhanced.astype(
        np.float32
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

    if mse == 0:
        return float("inf")

    max_pixel = 255.0

    psnr = 10 * np.log10(
        (max_pixel ** 2) / mse
    )

    return psnr


# ============================================================
# 8. UIQM - UICM
# ============================================================

def calculate_uicm(image):

    image = image.astype(
        np.float32
    )

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # RG and YB opponent color channels
    RG = R - G
    YB = 0.5 * (R + G) - B

    # Flatten
    RG = RG.flatten()
    YB = YB.flatten()

    # Mean
    mean_RG = np.mean(RG)
    mean_YB = np.mean(YB)

    # Standard deviation
    std_RG = np.std(RG)
    std_YB = np.std(YB)

    # Colorfulness component
    UICM = (
        -0.0268 * np.sqrt(
            mean_RG ** 2 +
            mean_YB ** 2
        )
        +
        0.1586 * np.sqrt(
            std_RG ** 2 +
            std_YB ** 2
        )
    )

    return UICM


# ============================================================
# 9. UIQM - UISM
# ============================================================

def calculate_uism(image):

    image = image.astype(
        np.float32
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    # Sobel gradients
    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Edge strength
    edge_strength = np.sqrt(
        sobel_x ** 2 +
        sobel_y ** 2
    )

    # Mean edge strength
    mean_edge = np.mean(
        edge_strength
    )

    # UISM
    UISM = mean_edge / 255.0

    return UISM


# ============================================================
# 10. UIQM - UIConM
# ============================================================

def calculate_uiconm(image):

    image = image.astype(
        np.float32
    )

    # Convert RGB → grayscale
    gray = cv2.cvtColor(
        image.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    # Local mean
    local_mean = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    # Local variance
    local_sq_mean = cv2.GaussianBlur(
        gray ** 2,
        (7, 7),
        0
    )

    local_variance = (
        local_sq_mean
        -
        local_mean ** 2
    )

    local_variance = np.maximum(
        local_variance,
        0
    )

    local_std = np.sqrt(
        local_variance
    )

    # Normalized contrast
    contrast = (
        local_std
        /
        (local_mean + 1e-6)
    )

    UIConM = np.mean(
        contrast
    )

    return UIConM


# ============================================================
# 11. UIQM
# ============================================================

def calculate_uiqm(image):

    # Calculate three components
    UICM = calculate_uicm(
        image
    )

    UISM = calculate_uism(
        image
    )

    UIConM = calculate_uiconm(
        image
    )

    # Standard UIQM weighting
    UIQM = (
        0.0282 * UICM
        +
        0.2953 * UISM
        +
        3.5753 * UIConM
    )

    return UIQM


# ============================================================
# 12. LOAD IMAGE
# ============================================================

print("=" * 70)
print("          RETINEX UNDERWATER IMAGE ENHANCEMENT")
print("=" * 70)


image = cv2.imread(
    INPUT_IMAGE
)


if image is None:

    print(
        "ERROR: Could not find",
        INPUT_IMAGE
    )

    exit()


print(
    "Image loaded:",
    INPUT_IMAGE
)


# BGR → RGB
original = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


original_float = original.astype(
    np.float32
)


# ============================================================
# 13. STEP 1 — SSR
# ============================================================

print("\n[1/3] Applying SSR...")


ssr_raw = np.zeros_like(
    original_float
)


for c in range(3):

    ssr_raw[:, :, c] = ssr(
        original_float[:, :, c],
        SSR_SIGMA
    )


ssr_output = normalize(
    ssr_raw
)


print("SSR completed.")


# ============================================================
# 14. STEP 2 — MSR
# ============================================================

print("\n[2/3] Applying MSR...")


# Your requested sequential pipeline:
# MSR receives SSR output

msr_raw = msr(
    ssr_output,
    scales=MSR_SCALES
)


msr_output = normalize(
    msr_raw
)


print("MSR completed.")


# ============================================================
# 15. STEP 3 — MSRCR
# ============================================================

print("\n[3/3] Applying MSRCR...")


# Your requested sequential pipeline:
# MSRCR receives MSR output

msrcr_raw = msrcr(
    msr_output,
    original_float,
    alpha=ALPHA,
    beta=BETA,
    gain=GAIN,
    offset=OFFSET
)


msrcr_output = normalize(
    msrcr_raw
)


print("MSRCR completed.")


# ============================================================
# 16. SAVE RESULTS
# ============================================================

cv2.imwrite(
    "SSR_result.png",
    cv2.cvtColor(
        ssr_output,
        cv2.COLOR_RGB2BGR
    )
)


cv2.imwrite(
    "MSR_result.png",
    cv2.cvtColor(
        msr_output,
        cv2.COLOR_RGB2BGR
    )
)


cv2.imwrite(
    "MSRCR_result.png",
    cv2.cvtColor(
        msrcr_output,
        cv2.COLOR_RGB2BGR
    )
)


print("\nResults saved:")
print("1. SSR_result.png")
print("2. MSR_result.png")
print("3. MSRCR_result.png")


# ============================================================
# 17. CALCULATE MSE, PSNR AND UIQM
# ============================================================

print("\n")
print("=" * 70)
print("             IMAGE QUALITY METRICS")
print("=" * 70)


# ------------------------------------------------------------
# Original
# ------------------------------------------------------------

original_mse = 0
original_psnr = float("inf")
original_uiqm = calculate_uiqm(
    original
)


# ------------------------------------------------------------
# SSR
# ------------------------------------------------------------

ssr_mse = calculate_mse(
    original,
    ssr_output
)

ssr_psnr = calculate_psnr(
    original,
    ssr_output
)

ssr_uiqm = calculate_uiqm(
    ssr_output
)


# ------------------------------------------------------------
# MSR
# ------------------------------------------------------------

msr_mse = calculate_mse(
    original,
    msr_output
)

msr_psnr = calculate_psnr(
    original,
    msr_output
)

msr_uiqm = calculate_uiqm(
    msr_output
)


# ------------------------------------------------------------
# MSRCR
# ------------------------------------------------------------

msrcr_mse = calculate_mse(
    original,
    msrcr_output
)

msrcr_psnr = calculate_psnr(
    original,
    msrcr_output
)

msrcr_uiqm = calculate_uiqm(
    msrcr_output
)


# ============================================================
# 18. PRINT METRICS TABLE
# ============================================================

print(
    f"{'Image':<15}"
    f"{'MSE':>15}"
    f"{'PSNR (dB)':>15}"
    f"{'UIQM':>15}"
)

print("-" * 60)


print(
    f"{'Original':<15}"
    f"{original_mse:>15.4f}"
    f"{'∞':>15}"
    f"{original_uiqm:>15.4f}"
)


print(
    f"{'SSR':<15}"
    f"{ssr_mse:>15.4f}"
    f"{ssr_psnr:>15.4f}"
    f"{ssr_uiqm:>15.4f}"
)


print(
    f"{'MSR':<15}"
    f"{msr_mse:>15.4f}"
    f"{msr_psnr:>15.4f}"
    f"{msr_uiqm:>15.4f}"
)


print(
    f"{'MSRCR':<15}"
    f"{msrcr_mse:>15.4f}"
    f"{msrcr_psnr:>15.4f}"
    f"{msrcr_uiqm:>15.4f}"
)


print("-" * 60)


# ============================================================
# 19. FIND BEST UIQM
# ============================================================

uiqm_values = {
    "SSR": ssr_uiqm,
    "MSR": msr_uiqm,
    "MSRCR": msrcr_uiqm
}


best_uiqm_method = max(
    uiqm_values,
    key=uiqm_values.get
)


print(
    "\nBest UIQM:",
    best_uiqm_method
)


# ============================================================
# 20. DISPLAY ALL RESULTS
# ============================================================

plt.figure(
    figsize=(20, 6)
)


# ------------------------------------------------------------
# Original
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    1
)

plt.imshow(
    original
)

plt.title(
    "Original Underwater"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# SSR
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    2
)

plt.imshow(
    ssr_output
)

plt.title(
    f"SSR\nUIQM = {ssr_uiqm:.3f}"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# MSR
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    3
)

plt.imshow(
    msr_output
)

plt.title(
    f"MSR ← SSR\nUIQM = {msr_uiqm:.3f}"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# MSRCR
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    4
)

plt.imshow(
    msrcr_output
)

plt.title(
    f"MSRCR ← MSR\nUIQM = {msrcr_uiqm:.3f}"
)

plt.axis(
    "off"
)


plt.tight_layout()

plt.show()


# ============================================================
# 21. PIPELINE
# ============================================================

print("\n" + "=" * 70)

print("PIPELINE")

print("=" * 70)

print("""
                 hazy.png
                    |
                    v
                   SSR
                    |
                    v
                SSR Result
                    |
                    v
                   MSR
                    |
                    v
                MSR Result
                    |
                    v
                  MSRCR
                    |
                    v
               Final Result
""")


# ============================================================
# 22. METRIC EXPLANATION
# ============================================================

print("=" * 70)
print("METRIC INTERPRETATION")
print("=" * 70)

print("""
MSE  : Lower is generally better.
PSNR : Higher is generally better.
UIQM : Higher generally indicates better underwater
       perceptual quality.

IMPORTANT:
MSE and PSNR here use the ORIGINAL HAZY IMAGE as reference.
Therefore, they measure how much the enhancement differs
from the input image, NOT how close it is to a true clear
ground-truth image.

For proper enhancement evaluation:
    Original Hazy Image
             |
             |---- Enhanced Image
             |
             +---- Ground Truth / Clear Image
                         |
                         v
                    MSE + PSNR

UIQM does not require a ground-truth image.
""")