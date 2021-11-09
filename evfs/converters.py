import random
from math import floor
import time
from random import choices
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from .mask_converter import *
from .misc import *
from .uniform_crossover import *
from .crossb_mutate import *                  
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
    f1_score,
    accuracy_score,
)



def bin_to_real(bits, lower_limit, upper_limit, num):

        """

         This function converts binary number to real number according to this equation

                     m = a +( b − a/2 k − 1) · m 

         paper link = "https://www.researchgate.net/publication/277604645_Genetic_Algorithm_using_Theory_of_Chaos"

         ...

         Attributes
         ----------

        bits:List
            binary bits in the form of list

        lower_limit:Int
            lower limit for binary bits

        upper_limit:Int
            upper_limit for binary bits

         num:Int
             m* for the equation

         Returns
         -------

         int:
             returns from the equation



        """

        up = upper_limit - lower_limit

        down = 2 ** len(bits) - 1

        return lower_limit + ((up / down) * num)


def masktodecimal(upper_limit, lower_limit, num, k):

        """

        This function converts mask to decimal

        using equation  m = a +( b − a/2 k − 1) · m 

        paper link = "https://www.researchgate.net/publication/277604645_Genetic_Algorithm_using_Theory_of_Chaos"

        ...

        Attributes
        ----------

        lower_limit:int
           lower limit for binary bits

        upper_limit:int
           upper_limit for binary bits

        num:int
            defined as m* in the equation

        k:int
            defined as k in the equation

        Returns
        -------

         int:
            decimal number

        """

        up = (num - lower_limit) * (2 ** k - 1)

        down = upper_limit - lower_limit
        
        return up / down
