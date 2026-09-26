
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# # ============================================================
# # SETTINGS
# # ============================================================

# INPUT_IMAGE = "hazy.png"  # Input image filename

# # MSR scales
# SCALES = [15, 80, 250]

# # SSR scale
# SSR_SIGMA = 80

# # MSRCR parameters
# ALPHA = 125
# BETA = 46

# # Gain and offset
# GAIN = 5.0
# OFFSET = -25.0


# # ============================================================
# # 1. SINGLE SCALE RETINEX (SSR)
# # ============================================================

# def ssr(channel, sigma):
#     """
#     Single Scale Retinex

#     R(x,y) = log(I(x,y)) - log(F(x,y) * I(x,y))

#     channel : one image channel
#     sigma   : Gaussian scale
#     """

#     # Convert to float
#     channel = channel.astype(np.float32)

#     # Avoid log(0)
#     channel = np.maximum(channel, 1.0)

#     # Estimate surrounding illumination
#     blur = cv2.GaussianBlur(
#         channel,
#         (0, 0),
#         sigmaX=sigma,
#         sigmaY=sigma
#     )

#     # Avoid log(0)
#     blur = np.maximum(blur, 1.0)

#     # Retinex
#     result = np.log(channel) - np.log(blur)

#     return result


# # ============================================================
# # 2. MULTI SCALE RETINEX (MSR)
# # ============================================================

# def msr(image, scales=SCALES):
#     """
#     Multi Scale Retinex

#     MSR = average of SSR results
#     at multiple scales.
#     """

#     image = image.astype(np.float32)

#     # Output
#     result = np.zeros_like(
#         image,
#         dtype=np.float32
#     )

#     # Process each RGB channel
#     for c in range(3):

#         channel = image[:, :, c]

#         channel_result = np.zeros_like(
#             channel,
#             dtype=np.float32
#         )

#         # Apply SSR for every scale
#         for sigma in scales:

#             channel_result += ssr(
#                 channel,
#                 sigma
#             )

#         # Equal weights
#         channel_result /= len(scales)

#         result[:, :, c] = channel_result

#     return result


# # ============================================================
# # 3. COLOR RESTORATION
# # ============================================================

# def color_restoration(
#     image,
#     alpha=ALPHA,
#     beta=BETA
# ):
#     """
#     Color Restoration Function

#     C_i = beta * [
#             log(alpha * I_i)
#             - log(sum(I))
#           ]
#     """

#     image = image.astype(np.float32)

#     # Avoid zero
#     image = np.maximum(
#         image,
#         1.0
#     )

#     # R + G + B
#     channel_sum = np.sum(
#         image,
#         axis=2,
#         keepdims=True
#     )

#     # Avoid log(0)
#     channel_sum = np.maximum(
#         channel_sum,
#         1.0
#     )

#     # Color restoration
#     color = beta * (
#         np.log(alpha * image)
#         -
#         np.log(channel_sum)
#     )

#     return color


# # ============================================================
# # 4. MSRCR
# # ============================================================

# def msrcr(
#     image,
#     scales=SCALES,
#     alpha=ALPHA,
#     beta=BETA,
#     gain=GAIN,
#     offset=OFFSET
# ):
#     """
#     Multi Scale Retinex with Color Restoration

#     MSRCR =
#         G * (ColorRestoration * MSR + b)
#     """

#     # --------------------------------------------------------
#     # Step 1: MSR
#     # --------------------------------------------------------

#     msr_result = msr(
#         image,
#         scales=scales
#     )

#     # --------------------------------------------------------
#     # Step 2: Color restoration
#     # --------------------------------------------------------

#     color = color_restoration(
#         image,
#         alpha=alpha,
#         beta=beta
#     )

#     # --------------------------------------------------------
#     # Step 3: Combine
#     # --------------------------------------------------------

#     result = color * msr_result

#     # --------------------------------------------------------
#     # Step 4: Gain and offset
#     # --------------------------------------------------------

#     result = gain * (
#         result + offset
#     )

#     return result


# # ============================================================
# # 5. PERCENTILE NORMALIZATION
# # ============================================================

# def normalize_percentile(
#     image,
#     low_percentile=1,
#     high_percentile=99
# ):
#     """
#     Better normalization than simple min-max.

#     Removes extreme values before mapping
#     the image to 0-255.
#     """

#     image = image.astype(
#         np.float32
#     )

#     output = np.zeros_like(
#         image,
#         dtype=np.float32
#     )

#     # Process each channel separately
#     for c in range(3):

#         channel = image[:, :, c]

#         low = np.percentile(
#             channel,
#             low_percentile
#         )

#         high = np.percentile(
#             channel,
#             high_percentile
#         )

#         # Prevent division by zero
#         if high - low < 1e-6:

#             output[:, :, c] = 0

#             continue

#         # Clip extreme values
#         channel = np.clip(
#             channel,
#             low,
#             high
#         )

#         # Convert to 0-255
#         channel = (
#             (channel - low)
#             /
#             (high - low)
#             * 255.0
#         )

#         output[:, :, c] = channel

#     return np.clip(
#         output,
#         0,
#         255
#     ).astype(np.uint8)


# # ============================================================
# # 6. SIMPLE NORMALIZATION
# # ============================================================

# def normalize_global(image):
#     """
#     Global normalization.
#     Useful for checking raw Retinex output.
#     """

#     image = image.astype(
#         np.float32
#     )

#     min_value = image.min()
#     max_value = image.max()

#     if max_value - min_value < 1e-6:

#         return np.zeros_like(
#             image,
#             dtype=np.uint8
#         )

#     result = (
#         (image - min_value)
#         /
#         (max_value - min_value)
#         * 255.0
#     )

#     return np.clip(
#         result,
#         0,
#         255
#     ).astype(np.uint8)


# # ============================================================
# # 7. MSE CALCULATION
# # ============================================================

# def calculate_mse(
#     original,
#     enhanced
# ):
#     """
#     Mean Squared Error (MSE)

#     Lower MSE = less pixel-level error.
#     """

#     # Convert both images to float
#     original = original.astype(
#         np.float64
#     )

#     enhanced = enhanced.astype(
#         np.float64
#     )

#     # MSE formula
#     mse = np.mean(
#         (original - enhanced) ** 2
#     )

#     return mse


# # ============================================================
# # 8. PSNR CALCULATION
# # ============================================================

# def calculate_psnr(
#     original,
#     enhanced
# ):
#     """
#     Peak Signal-to-Noise Ratio (PSNR)

#     PSNR = 10 * log10(MAX_PIXEL^2 / MSE)

#     Higher PSNR = images are more similar.
#     """

#     # Calculate MSE
#     mse = calculate_mse(
#         original,
#         enhanced
#     )

#     # Identical images
#     if mse < 1e-10:

#         return float("inf")

#     # Maximum possible pixel value
#     MAX_PIXEL = 255.0

#     # PSNR formula
#     psnr = 10 * np.log10(
#         (MAX_PIXEL ** 2) / mse
#     )

#     return psnr


# # ============================================================
# # 9. LOAD IMAGE
# # ============================================================

# print("=" * 65)
# print("       UNDERWATER RETINEX IMAGE ENHANCEMENT")
# print("=" * 65)

# print(
#     "\nLoading:",
#     INPUT_IMAGE
# )

# image = cv2.imread(
#     INPUT_IMAGE
# )


# # ============================================================
# # 10. CHECK IMAGE
# # ============================================================

# if image is None:

#     print(
#         "\nERROR: Image could not be loaded."
#     )

#     print(
#         "\nMake sure:"
#     )

#     print(
#         "1. test_p6_.jpg exists"
#     )

#     print(
#         "2. It is in the same folder as this Python file"
#     )

#     print(
#         "3. The filename is exactly test_p6_.jpg"
#     )

#     exit()


# print(
#     "Image loaded successfully"
# )


# # ============================================================
# # 11. BGR → RGB
# # ============================================================

# image_rgb = cv2.cvtColor(
#     image,
#     cv2.COLOR_BGR2RGB
# )

# print(
#     "Image size:",
#     image_rgb.shape[1],
#     "x",
#     image_rgb.shape[0]
# )


# # ============================================================
# # 12. SSR
# # ============================================================

# print(
#     "\n[1/3] Running SSR..."
# )

# ssr_result = np.zeros_like(
#     image_rgb,
#     dtype=np.float32
# )

# for c in range(3):

#     ssr_result[:, :, c] = ssr(
#         image_rgb[:, :, c],
#         SSR_SIGMA
#     )

# print(
#     "SSR completed"
# )


# # ============================================================
# # 13. MSR
# # ============================================================

# print(
#     "\n[2/3] Running MSR..."
# )

# msr_result = msr(
#     image_rgb,
#     scales=SCALES
# )

# print(
#     "MSR completed"
# )


# # ============================================================
# # 14. MSRCR
# # ============================================================

# print(
#     "\n[3/3] Running MSRCR..."
# )

# msrcr_result = msrcr(
#     image_rgb,
#     scales=SCALES,
#     alpha=ALPHA,
#     beta=BETA,
#     gain=GAIN,
#     offset=OFFSET
# )

# print(
#     "MSRCR completed"
# )


# # ============================================================
# # 15. PRINT RAW RANGES
# # ============================================================

# print(
#     "\n" + "-" * 65
# )

# print(
#     "RAW OUTPUT RANGES"
# )

# print(
#     "-" * 65
# )

# print(
#     "SSR   :",
#     ssr_result.min(),
#     "to",
#     ssr_result.max()
# )

# print(
#     "MSR   :",
#     msr_result.min(),
#     "to",
#     msr_result.max()
# )

# print(
#     "MSRCR :",
#     msrcr_result.min(),
#     "to",
#     msrcr_result.max()
# )


# # ============================================================
# # 16. NORMALIZE OUTPUTS
# # ============================================================

# print(
#     "\nNormalizing outputs..."
# )

# ssr_display = normalize_percentile(
#     ssr_result
# )

# msr_display = normalize_percentile(
#     msr_result
# )

# msrcr_display = normalize_percentile(
#     msrcr_result
# )


# # ============================================================
# # 17. MSE + PSNR CALCULATION
# # ============================================================

# print(
#     "\n" + "-" * 65
# )

# print(
#     "MSE AND PSNR RESULTS"
# )

# print(
#     "-" * 65
# )


# # ------------------------------------------------------------
# # Original image as reference
# # ------------------------------------------------------------

# original_image = image_rgb


# # ------------------------------------------------------------
# # SSR MSE + PSNR
# # ------------------------------------------------------------

# mse_ssr = calculate_mse(
#     original_image,
#     ssr_display
# )

# psnr_ssr = calculate_psnr(
#     original_image,
#     ssr_display
# )


# # ------------------------------------------------------------
# # MSR MSE + PSNR
# # ------------------------------------------------------------

# mse_msr = calculate_mse(
#     original_image,
#     msr_display
# )

# psnr_msr = calculate_psnr(
#     original_image,
#     msr_display
# )


# # ------------------------------------------------------------
# # MSRCR MSE + PSNR
# # ------------------------------------------------------------

# mse_msrcr = calculate_mse(
#     original_image,
#     msrcr_display
# )

# psnr_msrcr = calculate_psnr(
#     original_image,
#     msrcr_display
# )


# # ============================================================
# # 18. PRINT MSE + PSNR
# # ============================================================

# print()

# print(
#     f"SSR   : MSE = {mse_ssr:.2f}   |   "
#     f"PSNR = {psnr_ssr:.2f} dB"
# )

# print(
#     f"MSR   : MSE = {mse_msr:.2f}   |   "
#     f"PSNR = {psnr_msr:.2f} dB"
# )

# print(
#     f"MSRCR : MSE = {mse_msrcr:.2f}   |   "
#     f"PSNR = {psnr_msrcr:.2f} dB"
# )


# # ============================================================
# # 19. SAVE RESULTS
# # ============================================================

# print(
#     "\nSaving results..."
# )


# # ------------------------------------------------------------
# # Original
# # ------------------------------------------------------------

# cv2.imwrite(
#     "Original.png",
#     image
# )


# # ------------------------------------------------------------
# # SSR
# # ------------------------------------------------------------

# cv2.imwrite(
#     "SSR_result.png",
#     cv2.cvtColor(
#         ssr_display,
#         cv2.COLOR_RGB2BGR
#     )
# )


# # ------------------------------------------------------------
# # MSR
# # ------------------------------------------------------------

# cv2.imwrite(
#     "MSR_result.png",
#     cv2.cvtColor(
#         msr_display,
#         cv2.COLOR_RGB2BGR
#     )
# )


# # ------------------------------------------------------------
# # MSRCR
# # ------------------------------------------------------------

# cv2.imwrite(
#     "MSRCR_result.png",
#     cv2.cvtColor(
#         msrcr_display,
#         cv2.COLOR_RGB2BGR
#     )
# )


# print(
#     "Original.png"
# )

# print(
#     "SSR_result.png"
# )

# print(
#     "MSR_result.png"
# )

# print(
#     "MSRCR_result.png"
# )


# # ============================================================
# # 20. DISPLAY RESULTS
# # ============================================================

# plt.figure(
#     figsize=(18, 5)
# )


# # ------------------------------------------------------------
# # Original
# # ------------------------------------------------------------

# plt.subplot(
#     1,
#     4,
#     1
# )

# plt.imshow(
#     image_rgb
# )

# plt.title(
#     "Original Underwater",
#     fontsize=13
# )

# plt.axis(
#     "off"
# )


# # ------------------------------------------------------------
# # SSR
# # ------------------------------------------------------------

# plt.subplot(
#     1,
#     4,
#     2
# )

# plt.imshow(
#     ssr_display
# )

# plt.title(
#     f"SSR (σ = 80)\n"
#     f"MSE = {mse_ssr:.2f}\n"
#     f"PSNR = {psnr_ssr:.2f} dB",
#     fontsize=11
# )

# plt.axis(
#     "off"
# )


# # ------------------------------------------------------------
# # MSR
# # ------------------------------------------------------------

# plt.subplot(
#     1,
#     4,
#     3
# )

# plt.imshow(
#     msr_display
# )

# plt.title(
#     f"MSR (15, 80, 250)\n"
#     f"MSE = {mse_msr:.2f}\n"
#     f"PSNR = {psnr_msr:.2f} dB",
#     fontsize=11
# )

# plt.axis(
#     "off"
# )


# # ------------------------------------------------------------
# # MSRCR
# # ------------------------------------------------------------

# plt.subplot(
#     1,
#     4,
#     4
# )

# plt.imshow(
#     msrcr_display
# )

# plt.title(
#     f"MSRCR\n"
#     f"MSE = {mse_msrcr:.2f}\n"
#     f"PSNR = {psnr_msrcr:.2f} dB",
#     fontsize=11
# )

# plt.axis(
#     "off"
# )


# plt.tight_layout()

# plt.show()


# # ============================================================
# # 21. FINAL MESSAGE
# # ============================================================

# print(
#     "\n" + "=" * 65
# )

# print(
#     "                 PROCESSING COMPLETED"
# )

# print(
#     "=" * 65
# )

# print(
#     "\nPipeline:"
# )

# print()

# print(
#     "Underwater Image"
# )

# print(
#     "       ↓"
# )

# print(
#     "      SSR"
# )

# print(
#     "       ↓"
# )

# print(
#     "      MSR"
# )

# print(
#     "       ↓"
# )

# print(
#     "     MSRCR"
# )

# print(
#     "       ↓"
# )

# print(
#     " Enhanced Image"
# )

# print()

# print(
#     "MSE and PSNR calculated for:"
# )

# print(
#     "• SSR"
# )

# print(
#     "• MSR"
# )

# print(
#     "• MSRCR"
# )

# print()

# print(
#     "Output files:"
# )

# print(
#     "• Original.png"
# )

# print(
#     "• SSR_result.png"
# )

# print(
#     "• MSR_result.png"
# )

# print(
#     "• MSRCR_result.png"
# )

# ----------------------------------newwwwwwww----------
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim


# ============================================================
# SETTINGS
# ============================================================

INPUT_IMAGE = "test_p12_.jpg"  # Input image filename

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
# 1. SINGLE SCALE RETINEX (SSR)
# ============================================================

def ssr(channel, sigma):
    """
    Single Scale Retinex

    R(x,y) = log(I(x,y)) - log(F(x,y) * I(x,y))
    """

    channel = channel.astype(np.float32)

    # Avoid log(0)
    channel = np.maximum(
        channel,
        1.0
    )

    # Gaussian illumination estimate
    blur = cv2.GaussianBlur(
        channel,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    # Avoid log(0)
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
    """
    Multi Scale Retinex

    MSR = average of SSR results
    at multiple scales.
    """

    image = image.astype(
        np.float32
    )

    result = np.zeros_like(
        image,
        dtype=np.float32
    )

    # Process each RGB channel
    for c in range(3):

        channel = image[:, :, c]

        channel_result = np.zeros_like(
            channel,
            dtype=np.float32
        )

        # Apply SSR at every scale
        for sigma in scales:

            channel_result += ssr(
                channel,
                sigma
            )

        # Equal weights
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
    """
    Color Restoration Function

    C_i = beta * [
            log(alpha * I_i)
            - log(sum(I))
          ]
    """

    image = image.astype(
        np.float32
    )

    # Avoid zero
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

    # Avoid log(0)
    channel_sum = np.maximum(
        channel_sum,
        1.0
    )

    # Color restoration
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
    """
    Multi Scale Retinex with Color Restoration

    MSRCR =
        G * (ColorRestoration * MSR + b)
    """

    # --------------------------------------------------------
    # Step 1: MSR
    # --------------------------------------------------------

    msr_result = msr(
        image,
        scales=scales
    )

    # --------------------------------------------------------
    # Step 2: Color restoration
    # --------------------------------------------------------

    color = color_restoration(
        image,
        alpha=alpha,
        beta=beta
    )

    # --------------------------------------------------------
    # Step 3: Combine
    # --------------------------------------------------------

    result = (
        color *
        msr_result
    )

    # --------------------------------------------------------
    # Step 4: Gain + offset
    # --------------------------------------------------------

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
    """
    Percentile normalization.

    Removes extreme values before mapping
    image to 0-255.
    """

    image = image.astype(
        np.float32
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    # Process each channel separately
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

        # Avoid division by zero
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
            * 255.0
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
# 6. SIMPLE NORMALIZATION
# ============================================================

def normalize_global(image):
    """
    Global normalization.
    """

    image = image.astype(
        np.float32
    )

    min_value = image.min()
    max_value = image.max()

    if max_value - min_value < 1e-6:

        return np.zeros_like(
            image,
            dtype=np.uint8
        )

    result = (
        (image - min_value)
        /
        (max_value - min_value)
        * 255.0
    )

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# 7. MSE
# ============================================================

def calculate_mse(
    original,
    enhanced
):
    """
    Mean Squared Error.

    Lower MSE = less pixel-level difference.
    """

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
# 8. PSNR
# ============================================================

def calculate_psnr(
    original,
    enhanced
):
    """
    Peak Signal-to-Noise Ratio.

    Higher PSNR = more similarity.
    """

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
# 9. SSIM
# ============================================================

def calculate_ssim(
    original,
    enhanced
):
    """
    Structural Similarity Index.

    Higher SSIM = higher structural similarity.
    """

    # RGB → grayscale
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
# 10. UIQM - UICM
# ============================================================

def calculate_uicm(image):
    """
    Underwater Image Colorfulness Measure.

    Measures color distortion/colorfulness.
    """

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

    # Sort values
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

    # Means
    mean_RG = np.mean(RG)
    mean_YB = np.mean(YB)

    # Variances
    var_RG = np.mean(
        (RG - mean_RG) ** 2
    )

    var_YB = np.mean(
        (YB - mean_YB) ** 2
    )

    # UICM
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
# 11. UIQM - EME
# ============================================================

def calculate_eme(
    channel,
    block_size=8
):
    """
    Enhancement Measure Estimation.

    Used as part of UISM.
    """

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
# 12. UIQM - UISM
# ============================================================

def calculate_uism(image):
    """
    Underwater Image Sharpness Measure.

    Measures sharpness/detail.
    """

    image = image.astype(
        np.float64
    )

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Sobel edge detection
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

    # Absolute edge magnitude
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

    # RGB weighted sharpness
    uism = (
        0.299 * R_eme
        +
        0.587 * G_eme
        +
        0.114 * B_eme
    )

    return uism


# ============================================================
# 13. UIQM - UIConM
# ============================================================

def calculate_uiconm(image):
    """
    Underwater Image Contrast Measure.

    Measures image contrast.
    """

    image = image.astype(
        np.uint8
    )

    # RGB → grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Standard deviation based contrast
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
# 14. COMPLETE UIQM
# ============================================================

def calculate_uiqm(image):
    """
    Underwater Image Quality Measure.

    UIQM =
        c1 * UICM
        +
        c2 * UISM
        +
        c3 * UIConM
    """

    # Standard coefficients
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    # Calculate components
    uicm = calculate_uicm(
        image
    )

    uism = calculate_uism(
        image
    )

    uiconm = calculate_uiconm(
        image
    )

    # Final UIQM
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
# 15. LOAD IMAGE
# ============================================================

print(
    "=" * 70
)

print(
    "          UNDERWATER RETINEX IMAGE ENHANCEMENT"
)

print(
    "=" * 70
)

print(
    "\nLoading:",
    INPUT_IMAGE
)

image = cv2.imread(
    INPUT_IMAGE
)


# ============================================================
# 16. CHECK IMAGE
# ============================================================

if image is None:

    print(
        "\nERROR: Image could not be loaded."
    )

    print(
        "\nMake sure:"
    )

    print(
        "1. test_p6_.jpg exists"
    )

    print(
        "2. It is in the same folder as this Python file"
    )

    print(
        "3. Filename is exactly test_p6_.jpg"
    )

    exit()


print(
    "Image loaded successfully"
)


# ============================================================
# 17. BGR → RGB
# ============================================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

print(
    "Image size:",
    image_rgb.shape[1],
    "x",
    image_rgb.shape[0]
)


# ============================================================
# 18. SSR
# ============================================================

print(
    "\n[1/3] Running SSR..."
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

print(
    "SSR completed"
)


# ============================================================
# 19. MSR
# ============================================================

print(
    "\n[2/3] Running MSR..."
)

msr_result = msr(
    image_rgb,
    scales=SCALES
)

print(
    "MSR completed"
)


# ============================================================
# 20. MSRCR
# ============================================================

print(
    "\n[3/3] Running MSRCR..."
)

msrcr_result = msrcr(
    image_rgb,
    scales=SCALES,
    alpha=ALPHA,
    beta=BETA,
    gain=GAIN,
    offset=OFFSET
)

print(
    "MSRCR completed"
)


# ============================================================
# 21. RAW RANGES
# ============================================================

print(
    "\n" + "-" * 70
)

print(
    "RAW OUTPUT RANGES"
)

print(
    "-" * 70
)

print(
    "SSR   :",
    ssr_result.min(),
    "to",
    ssr_result.max()
)

print(
    "MSR   :",
    msr_result.min(),
    "to",
    msr_result.max()
)

print(
    "MSRCR :",
    msrcr_result.min(),
    "to",
    msrcr_result.max()
)


# ============================================================
# 22. NORMALIZE OUTPUTS
# ============================================================

print(
    "\nNormalizing outputs..."
)

ssr_display = normalize_percentile(
    ssr_result
)

msr_display = normalize_percentile(
    msr_result
)

msrcr_display = normalize_percentile(
    msrcr_result
)


# ============================================================
# 23. MSE + PSNR + SSIM
# ============================================================

print(
    "\n" + "-" * 70
)

print(
    "MSE + PSNR + SSIM RESULTS"
)

print(
    "-" * 70
)

# Original image as reference
original_image = image_rgb


# ------------------------------------------------------------
# SSR
# ------------------------------------------------------------

mse_ssr = calculate_mse(
    original_image,
    ssr_display
)

psnr_ssr = calculate_psnr(
    original_image,
    ssr_display
)

ssim_ssr = calculate_ssim(
    original_image,
    ssr_display
)


# ------------------------------------------------------------
# MSR
# ------------------------------------------------------------

mse_msr = calculate_mse(
    original_image,
    msr_display
)

psnr_msr = calculate_psnr(
    original_image,
    msr_display
)

ssim_msr = calculate_ssim(
    original_image,
    msr_display
)


# ------------------------------------------------------------
# MSRCR
# ------------------------------------------------------------

mse_msrcr = calculate_mse(
    original_image,
    msrcr_display
)

psnr_msrcr = calculate_psnr(
    original_image,
    msrcr_display
)

ssim_msrcr = calculate_ssim(
    original_image,
    msrcr_display
)


# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print()

print(
    f"SSR   : "
    f"MSE = {mse_ssr:.2f}   |   "
    f"PSNR = {psnr_ssr:.2f} dB   |   "
    f"SSIM = {ssim_ssr:.4f}"
)

print()

print(
    f"MSR   : "
    f"MSE = {mse_msr:.2f}   |   "
    f"PSNR = {psnr_msr:.2f} dB   |   "
    f"SSIM = {ssim_msr:.4f}"
)

print()

print(
    f"MSRCR : "
    f"MSE = {mse_msrcr:.2f}   |   "
    f"PSNR = {psnr_msrcr:.2f} dB   |   "
    f"SSIM = {ssim_msrcr:.4f}"
)


# ============================================================
# 24. UIQM
# ============================================================

print(
    "\n" + "-" * 70
)

print(
    "UIQM RESULTS"
)

print(
    "-" * 70
)


# ------------------------------------------------------------
# SSR UIQM
# ------------------------------------------------------------

uiqm_ssr, uicm_ssr, uism_ssr, uiconm_ssr = calculate_uiqm(
    ssr_display
)


# ------------------------------------------------------------
# MSR UIQM
# ------------------------------------------------------------

uiqm_msr, uicm_msr, uism_msr, uiconm_msr = calculate_uiqm(
    msr_display
)


# ------------------------------------------------------------
# MSRCR UIQM
# ------------------------------------------------------------

uiqm_msrcr, uicm_msrcr, uism_msrcr, uiconm_msrcr = calculate_uiqm(
    msrcr_display
)


# ------------------------------------------------------------
# Print SSR
# ------------------------------------------------------------

print()

print(
    f"SSR   : UIQM = {uiqm_ssr:.4f}"
)

print(
    f"        UICM = {uicm_ssr:.4f}"
)

print(
    f"        UISM = {uism_ssr:.4f}"
)

print(
    f"        UIConM = {uiconm_ssr:.4f}"
)


# ------------------------------------------------------------
# Print MSR
# ------------------------------------------------------------

print()

print(
    f"MSR   : UIQM = {uiqm_msr:.4f}"
)

print(
    f"        UICM = {uicm_msr:.4f}"
)

print(
    f"        UISM = {uism_msr:.4f}"
)

print(
    f"        UIConM = {uiconm_msr:.4f}"
)


# ------------------------------------------------------------
# Print MSRCR
# ------------------------------------------------------------

print()

print(
    f"MSRCR : UIQM = {uiqm_msrcr:.4f}"
)

print(
    f"        UICM = {uicm_msrcr:.4f}"
)

print(
    f"        UISM = {uism_msrcr:.4f}"
)

print(
    f"        UIConM = {uiconm_msrcr:.4f}"
)


# ============================================================
# 25. SAVE RESULTS
# ============================================================

print(
    "\nSaving results..."
)


# ------------------------------------------------------------
# Original
# ------------------------------------------------------------

cv2.imwrite(
    "Original.png",
    image
)


# ------------------------------------------------------------
# SSR
# ------------------------------------------------------------

cv2.imwrite(
    "SSR_result.png",
    cv2.cvtColor(
        ssr_display,
        cv2.COLOR_RGB2BGR
    )
)


# ------------------------------------------------------------
# MSR
# ------------------------------------------------------------

cv2.imwrite(
    "MSR_result.png",
    cv2.cvtColor(
        msr_display,
        cv2.COLOR_RGB2BGR
    )
)


# ------------------------------------------------------------
# MSRCR
# ------------------------------------------------------------

cv2.imwrite(
    "MSRCR_result.png",
    cv2.cvtColor(
        msrcr_display,
        cv2.COLOR_RGB2BGR
    )
)


print(
    "Original.png"
)

print(
    "SSR_result.png"
)

print(
    "MSR_result.png"
)

print(
    "MSRCR_result.png"
)


# ============================================================
# 26. DISPLAY RESULTS
# ============================================================

plt.figure(
    figsize=(22, 6)
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
    image_rgb
)

plt.title(
    "Original Underwater",
    fontsize=13
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
    ssr_display
)

plt.title(
    f"SSR (σ = 80)\n"
    f"PSNR = {psnr_ssr:.2f} dB\n"
    f"SSIM = {ssim_ssr:.4f}\n"
    f"UIQM = {uiqm_ssr:.4f}",
    fontsize=10
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
    msr_display
)

plt.title(
    f"MSR (15, 80, 250)\n"
    f"PSNR = {psnr_msr:.2f} dB\n"
    f"SSIM = {ssim_msr:.4f}\n"
    f"UIQM = {uiqm_msr:.4f}",
    fontsize=10
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
    msrcr_display
)

plt.title(
    f"MSRCR\n"
    f"PSNR = {psnr_msrcr:.2f} dB\n"
    f"SSIM = {ssim_msrcr:.4f}\n"
    f"UIQM = {uiqm_msrcr:.4f}",
    fontsize=10
)

plt.axis(
    "off"
)


plt.tight_layout()


# ============================================================
# 27. SAVE COMPARISON
# ============================================================

comparison_path = (
    "Retinex_Comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=150,
    bbox_inches="tight"
)

print(
    "\nComparison saved as:"
)

print(
    comparison_path
)


# ============================================================
# 28. SHOW
# ============================================================

plt.show()


# ============================================================
# 29. FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "                 PROCESSING COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nPipeline:"
)

print(
    "Underwater Image"
)

print(
    "       ↓"
)

print(
    "      SSR"
)

print(
    "       ↓"
)

print(
    "      MSR"
)

print(
    "       ↓"
)

print(
    "     MSRCR"
)

print(
    "       ↓"
)

print(
    " Enhanced Images"
)

print()

print(
    "Evaluation Metrics:"
)

print(
    "• MSE"
)

print(
    "• PSNR"
)

print(
    "• SSIM"
)

print(
    "• UIQM"
)

print(
    "• UICM"
)

print(
    "• UISM"
)

print(
    "• UIConM"
)

print()

print(
    "Done!"
)