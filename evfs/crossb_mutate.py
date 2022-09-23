import random
from math import floor
import time
from random import choices
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .mask_converter import *
from .misc import *
from .uniform_crossover import *
from .converters import *                 
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
    f1_score,
    accuracy_score,
)


def cross_breeding( parent1: list, parent2: list):

        """
        This function cross breeds the two parents by dividing in half and combining them


        ...

        Attributes
        ----------

        parent1:List
           features in vectorized form

        parent2:List
           features in vectorized form



        Returns
        -------

        :List
            new set of features resulted from crossbreeding

        """

        first_divide = parent1[: len(parent1) // 2]

        secodn_divide = parent2[len(parent2) // 2 :]

        return first_divide + secodn_divide

def combo_cross_breeding( parent1: list, parent2: list):

        """

        This function cross breeds the two parents by joinining  and combining them
        ...

        Attributes
        ----------


        parent1:List
           features in vectorized form

        parent2:List
           features in vectorized form



        Returns
        -------

        :List
            new set of features resulted from randomcrossbreeding

        """
        final = []

        # random.seed(0)
        for i in range(len(parent1)):
            if parent1[i] == 0 and parent2[i] == 0:
                final.append(0)
            else:
                final.append(1)

        # first_divide=parent1[:index]

        # secodn_divide=parent2[index:]

        return final

def random_cross_breeding( parent1: list, parent2: list):

        """

        This function cross breeds the two parents by divinding it at a random index and combining them
        ...

        Attributes
        ----------


        parent1:List
           features in vectorized form

        parent2:List
           features in vectorized form



        Returns
        -------

        :List
            new set of features resulted from randomcrossbreeding

        """

        end_index = min(len(parent1), len(parent2))

        # random.seed(0)

        index = random.randint(1, end_index - 1)

        first_divide = parent1[:index]

        secodn_divide = parent2[index:]

        return first_divide + secodn_divide

def mutate(population: list):

        """

        This function mutates the creatures at a random Index


        Attributes
        ----------


        population :List
           features in vectorized form


        Returns
        -------

        population :List
            new set of features resulted from mutation

        """

        selected_index = choices(population)

        if population[selected_index[0]] == 1:
            population[selected_index[0]] = 0
        else:
            population[selected_index[0]] = 1
        return population