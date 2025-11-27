
from rgb2hsi import getHSIfromRGB
from hsi2dfclass import pred


def detectDF(image_path, return_hsi=False):
    """Run hyperspectral detection and optionally return the HSI cube."""
    hsi_fp32 = getHSIfromRGB(image_path)
    results = pred(hsi_fp32)
    if return_hsi:
        return results, hsi_fp32
    return results


# results = detectDF("test.png")
# print(results)
