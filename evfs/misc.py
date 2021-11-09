import random
from math import floor
import time
from random import choices
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from mask_converter import *
from misc import *
from uniform_crossover import *
from converters import *
from crossb_mutate import *                  
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
    f1_score,
    accuracy_score,
)



def find_input(binary_input: list,features):

        """

        This functions converts binary inputs to features for eval function


        Attributes
        ----------


        binary_input :list
           length for the mask



        Returns
        -------

        :list
            return gray mask

        """

        action = []

        for index in range(len(binary_input)):
            if binary_input[index] == 1:
                action.append(features[index])
        return action
    

    
def make_graph( x, y, title, yname):

        plt.plot(x, y)

        plt.xlabel("gen")

        plt.ylabel(yname)

        plt.title(title)

        plt.show()



def create_vector(indexes, singularity):

        """

        This function creates a one hot vector for given indexes

        ...

        Attributes
        ----------

        indexes :List
            list containing indexes to be vectorized i.e 1

        singularity:List
            encoded list with feature length

        Returns
        -------

        evolve :List
            encoded list with selected features



        """

        evolve = singularity.copy()
        for index in indexes:
            evolve[index] = 1
        return evolve

def tracker( selected_features,features):

        """

         This function is part of the tracking the evolution cycle of features

         ...

         Attributes
         ----------

        selected_features:List
             features in string of list form


         Returns
         -------

         vectorized :List
             vectorized features



        """

        indexes = []

        for f in selected_features:

            indexes.append(features.index(f))

        vectorized = create_vector(indexes, [0] * len(features))

        return vectorized
    
