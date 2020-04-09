"""
Author: Vipul
Date: 23/07/19
Status: Development
"""
import cv2
import numpy as np


def hist_equalization(img):
    """
    Method to perform contrast enhancement using histogram equalization

    :param img: input image
    :return: output histogram equalized image
    """
    equ = cv2.equalizeHist(img)
    return equ


def adjust_gamma(img, gamma=1.0):
    """
    Method to perform illumination adjustment

    :param img: input image
    :param gamma: factor to adjust illumination
                  default:1.0
                  for lighter: <1.0
                  for darker: >1.0
    Additionally, LUT means to find color in Look Up Table
    :return: dark or lighter output image
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def unsharp(image, sigma, strength):
    """
    Method to perform unsharpening mask

    :param image: input image
    :param sigma: integral factor to adjust blurring from the centre 3 or 5
    :param strength: integral factor to adjust sharpening
    :return: sharpened output image
    """
    image_mf = cv2.medianBlur(image, sigma)

    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf, cv2.CV_64F)

    # Calculate the sharpened image
    sharp = image - strength * lap

    # Saturate the pixels in either direction
    sharp[sharp > 255] = 255
    sharp[sharp < 0] = 0

    return sharp


def global_thresh(img):
    """
    Method to perform global thresholding

    :param img: input image
    :return: threshold image
    """
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return th1


def otsu_binarization(img):
    """
    Method to perform Otsu Binarization

    :param img: input image
    :return: thresholded image
    """
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


def gauss_otsu(img):
    """
    Method to perform gaussian blur with otsu binarization
    :param img: input image
    :return: thresholded image
    """

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th1


def clahe(img):
    """
    Method to perform Contrast Limiting Adaptive Histogram Equalization for colored images
    :param img: input image
    :return: adaptive threshold image
    """
    # Convert image to LAB color
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Split the channels of image
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L-channel
    clh = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clh.apply(l)
    # Merge with channels back
    limg = cv2.merge((cl1, a, b))
    # Convert the LAB model to RGB again
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

img = cv2.imread("data/image_8.jpg", 0)
res = adjust_gamma(img,0.3)
sharp = unsharp(res,3,2)
final = gauss_otsu(sharp)
#final = otsu_binarization(sharp)
cv2.imshow("Image", final)
cv2.waitKey(0)
cv2.destroyAllWindows()