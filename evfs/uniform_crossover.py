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
from .converters import *
from .crossb_mutate import *                  


def uniformCrossover(parent1: list, parent2: list, mask: list):

        """

        This function performs uniform crossover and mutation with the help of a mask



        ...

        Attributes
        ----------

        parent1:List
           features in vectorized form

        parent2:List
           features in vectorized form

        mask:string
           string of bits used to form unifrom crossover and mask



        Returns
        -------

        child :List
            new set of features resulted from uniform crossover and mutation



        """

        if parent2 == "mutate":

            child = []
            index = 0

            for bits in mask:
                if bits == "0" and parent1[index] == 0:

                    child.append(0)
                else:
                    child.append(1)
                index += 1
            return child

        else:
            child = []
            index = 0

            # sanity check if we get length error while converting somewhere in code

            smallest = len(parent1) != len(parent2)

            if smallest is True:

                if len(parent1) > len(parent2):
                    smallest_len = len(parent2)
                    large_parent = parent1
                else:
                    smallest_len = len(parent1)
                    large_parent = parent2

                for bits in mask:
                    if index >= smallest_len:
                        break
                    if bits == 0:
                        child.append(parent2[index])
                    else:
                        child.append(parent1[index])
                    index += 1

                child += large_parent[smallest_len:]
            else:

                for bits in mask:
                    if bits == 0:
                        child.append(parent2[index])
                    else:
                        child.append(parent1[index])
                    index += 1

            return child
