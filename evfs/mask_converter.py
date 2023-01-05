import random
from math import floor
import time
from random import choices
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .misc import *
from .uniform_crossover import *
from .converters import *
from .crossb_mutate import *
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
    f1_score,
    accuracy_score,
)


def dec_to_bin(num):

    """

    A function used to convert decimal to binary
    ...

    Attributes
    ----------
    num :
        decimal number

    Returns
    -------

    binary :
        binary form of decimal

    """

    return "{0:b}".format(int(num))


def chaos_equation(Lambda, xn):

    """

    A function used to convert decimal to binary
    ...

    Attributes
    ----------
    lambda :int
        lambda value used in chaos equation

    xn:int
        "https://en.wikipedia.org/wiki/Logistic_map"

    Returns
    -------

    predicted chaos :int
        used to get the future chaos value



    """

    return round(Lambda * (xn * (1 - xn)), 4)


def mask_converter(gray_mask: list, lambda_mask: list, features):
    """

    A function used to convert masks
    ...

    Attributes
    ----------
    gray_mask :List
        mask value in gray encoding

    lambda:List
        mask value in gray encoding

    Returns
    -------

    gray mask :list
        converted gray mask




    """

    binary_mask, binary_lambda_mask = gray_to_binary(
        gray_mask, features
    ), gray_to_binary(lambda_mask, features)

    if len(binary_mask) < len(features):

        """
        a little trick used to make the length of masks as same as features in future we need to change it

        """

        binary_mask += [0] * len(features) - len(binary_mask)  # error chance

    decimal, decimal_lambda = int(binary_mask, 2), int(binary_lambda_mask, 2)

    mask_interval, lambd = bin_to_real(binary_mask, 0, 1, decimal), bin_to_real(
        binary_lambda_mask, 0, 4, decimal_lambda
    )

    chaos = chaos_equation(round(lambd, 4), round(mask_interval, 4))

    modified_mask_dec = masktodecimal(1, 0, chaos, len(features))

    modified_mask_dec = round(modified_mask_dec, 4)

    mod_dec_mask = dec_to_bin(modified_mask_dec)

    return binary_to_gray(mod_dec_mask, features)


def xor_conv(x1, x2):

    if x1 == x2:
        return "0"
    return "1"


# Helper function to flip the bit
def flip(c):

    return "1" if (c == "0") else "0"


def binary_to_gray(binary, features):
    """

     This function converts binary to gray

     ...

     Attributes
     ----------

    binary:List
         binary  encoding


     Returns
     -------

     :List
         gray encoding




    """

    binary = "".join(map(str, binary))

    gray = ""

    gray += binary[0]

    for i in range(1, len(binary)):

        gray += xor_conv(binary[i - 1], binary[i])

    exten = [0] * (len(features) - len(gray))

    exten = "".join(map(str, exten))

    return gray + exten


def gray_to_binary(gray, features):

    """

     This function converts gray to binary

     ...

     Attributes
     ----------

    gray:List
         gray encoding


     Returns
     -------

     :List
         binary encoding




    """

    gray = "".join(map(str, gray))

    binary = ""

    binary += gray[0]

    # Compute remaining bits
    for i in range(1, len(gray)):

        if gray[i] == "0":
            binary += binary[i - 1]

        else:
            binary += flip(binary[i - 1])

    exten = [0] * (len(features) - len(binary))

    exten = "".join(map(str, exten))

    return binary + exten
