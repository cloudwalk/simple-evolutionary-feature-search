import random
from math import floor
import time
from random import choices
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
    f1_score,
    accuracy_score,
)
from .mask_converter import *
from .misc import *
from .uniform_crossover import *
from .converters import *
from .crossb_mutate import *                  


class EvolutionaryFeatureSelector:
    """
    A class used to represent Evolutionary algorithm based feature selection
    ...
    Attributes
    ----------
    generations : int
        Total number of generations needed for efs to run
    total_creatures : int
        random creatures for each generation to increase search space
    features : list
        list of features
    name: str
        dictionary name in which we store winners
    Returns
    -------
    features: list
        final winning features for the evaluation function
    winning_key:int
        observed value for the selected features
    innovation_dic:dictionary
        contains information related to every feature and its formation used for tracking the orgin of features
    vectorized:list
         winning features represented in [0,1] form
    """

    def __init__(self, generations, features, dicName, creaturesNumber):

        self.top_creatures = {}

        self.generations = generations

        self.total_creatures = creaturesNumber

        self.features = features

        self.name = dicName

        self.graph = []

        self.earlier_winners = []

        self.individual_winners = []

        self.binary_tracker = {}

        self.temp = {}
        
        self.dic_format = {
            "parent1": None,
            "parent2": None,
            "genecode": None,
            "gen": None,
            "mask": None,
            "lambda": None,
        }

        self.innovation_dic = {}

        self.innovation_num = 0

        self.time = []

        self.start = 0

        assert generations < len(features), "generations>=len(features)"

        
    def __chaotic_population(
        self, pop_size, dim_size, num_col, minf=0, maxf=None, shuffled=False
    ):

        """
        @!Param -> pop_size: Number of members in the initial population
        @!Param -> dim_size: Number of genes that will make a chromosome
        @!Param -> ini_lambd: A value between 0 and 1 (This parameter is extremely important because chaotic functions are highly sensitive to initial values)
        @!Param -> num_col: Total number of features in the dataset
        @!Param -> minf: Min number of features
        @!Param -> maxf: Max number of features (None)
        @!Param -> shuffled: Shuffled the list of indexes, avoiding repetition in the initial and final indexes of the chromosome
        @!Return -> A list containing an initial population
        """

        def boundary(position, size):
            if position < minf:
                return minf
            if position > size:
                return size
            return position

        if maxf is None:
            maxf = num_col

        chromossome_list = []
        for i in range(pop_size):
            sample = []
            temp_list = np.arange(num_col)

            if shuffled:
                np.random.shuffle(temp_list)

            lambd_list = np.zeros(dim_size)
            ini_lambd = np.random.uniform(low=0, high=1)

            lambd_list[0] = 4 * ini_lambd * (1 - ini_lambd)
            for i in range(1, dim_size):
                lambd_list[i] = 4 * lambd_list[i - 1] * (1 - lambd_list[i - 1])

            R = 2 * (lambd_list - 0.5)

            # print(R)
            for idx in range(dim_size):
                asw_idx = floor(minf + R[idx] * (maxf - minf))
                asw_idx = boundary(asw_idx, (len(temp_list) - 1))
                sample = np.append(sample, temp_list[asw_idx])

                temp_list = np.delete(temp_list, [asw_idx])
            chromossome_list.append(np.sort(sample))

        return np.array(chromossome_list).astype(np.int)



    def __mutate_winners(self, population: list):

        """
        This function mutates the creatures with respect to winners from earlier generations
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

        individual = choices(self.individual_winners)

        on = self.features.index(individual[0])

        population[on] = 1

        if population[selected_index[0]] == 1:

            population[selected_index[0]] = 0

        else:
            population[selected_index[0]] = 1

        return population

    def __dictionary_key_finder(self, dictionary, genecode):

        """
        This function is used to find keys in dictionary for particular genecode
        Attributes
        ----------
        dictionary :dict
           dictionary we need to find key in
        genecode:list
            list used to find key
        Returns
        -------
        key :int
            the key value for a particular genecode
        """
        for key, value in dictionary.items():
            if value["genecode"] == genecode:
                return key

    def __dictionary_append(self, parent1_value, parent2_value, genecode, gen):

        """
        This function is used to store values in dictionary
        Attributes
        ----------
        parent1_value :List
           features in vectorized form
        parent2_value:List
            features in vectorized form
        genecode:List
            genecode to be appended in the dictionary
        gen:int
            generation in which it was formed
        Returns
        -------
        None
        """

        self.innovation_dic[self.innovation_num] = self.dic_format.copy()

        self.innovation_dic[self.innovation_num][
            "parent1"
        ] = self.__dictionary_key_finder(self.innovation_dic, parent1_value)

        if parent2_value != "mutated":

            self.innovation_dic[self.innovation_num][
                "parent2"
            ] = self.__dictionary_key_finder(self.innovation_dic, parent2_value)

        self.innovation_dic[self.innovation_num]["genecode"] = genecode

        self.innovation_dic[self.innovation_num]["gen"] = gen

        self.innovation_dic[self.innovation_num]["mask"] = self.innovation_dic[
            self.innovation_dic[self.innovation_num]["parent1"]
        ]["mask"]

        self.innovation_dic[self.innovation_num]["lambda"] = self.innovation_dic[
            self.innovation_dic[self.innovation_num]["parent1"]
        ]["mask"]

        self.innovation_num += 1

    def __dictionary_append_mask(self, parent1_value, parent2_value, genecode, gen, mask):

        """
        This function is used to store values in dictionary
        Attributes
        ----------
        parent1_value :List
           features in vectorized form
        parent2_value:List
            features in vectorized form
        genecode:List
            genecode to be appended in the dictionary
        gen:int
            generation in which it was formed
        mask:
            mask used to find new mask
        Returns
        -------
        None
        """

        self.innovation_dic[self.innovation_num] = self.dic_format.copy()

        temp = self.__dictionary_key_finder(self.innovation_dic, parent1_value)

        self.innovation_dic[self.innovation_num]["parent1"] = temp

        if parent2_value != "mutated":

            self.innovation_dic[self.innovation_num][
                "parent2"
            ] = self.__dictionary_key_finder(self.innovation_dic, parent2_value)
        else:
            self.innovation_dic[self.innovation_num]["parent2"] = "mutatedchaos"

        self.innovation_dic[self.innovation_num]["genecode"] = genecode

        self.innovation_dic[self.innovation_num]["gen"] = gen

        self.innovation_dic[self.innovation_num]["lambda"] = self.innovation_dic[temp][
            "lambda"
        ]

        gray_mask = "".join(map(str, mask))

        lambda_mask = self.innovation_dic[temp]["lambda"]

        new_lambda = "".join(map(str, lambda_mask))

        #new_mask = self.mask_converter(gray_mask, lambda_mask)
        new_mask = mask_converter(gray_mask, lambda_mask,self.features)
        self.innovation_dic[self.innovation_num]["mask"] = new_mask

        self.innovation_dic[self.innovation_num]["lambda"] = new_lambda

        self.innovation_num += 1

    def __dictionary_append_mutate(self, parent, gen):

        """
        This function is used to store values in dictionary
        Attributes
        ----------
        parent:List
           features in vectorized form
        gen:int
            generation in which it was formed
        Returns
        -------
        None
        """

        self.innovation_dic[self.innovation_num] = self.dic_format.copy()

        self.innovation_dic[self.innovation_num][
            "parent1"
        ] = self.__dictionary_key_finder(self.innovation_dic, parent)

        self.innovation_dic[self.innovation_num]["parent2"] = "mutated"

        genecode = self.mutate(parent.copy())

        self.innovation_dic[self.innovation_num]["genecode"] = genecode

        self.innovation_dic[self.innovation_num]["gen"] = gen

        self.innovation_num += 1

        return parent

    def __create_mask(self, length,features):

        """
        This function is used to create mask
        Attributes
        ----------
        length :int
           length for the mask
        Returns
        -------
        :list
            return gray mask
        """

        mask = []
        for i in range(length):
            mask.append(random.randint(0, 1))
        #return self.binary_to_gray(mask)
        return binary_to_gray(mask,features)

    def __select_winners_top(self, array: list, gen, evalFunction, variables):

        """
        This function selects the top winners from the creatures pool
        Attributes
        ----------
        array :List
           contains feature that needs to be evaluated
        Returns
        -------
        winners:list
            returns list of winners
        """

        winners = []

        final = float("-inf")

        feat = []

        dic = {}

        for creature in array:

            input = find_input(creature,self.features)

            if len(input) == 0:
                # print(creature,"creature error!!!!!")
                continue

            evaluate = evalFunction(input)

            score = evaluate.func(*variables, gen)

            dic[score] = creature

            if final < score:

                final = score

                feat = input

        dic = dict(sorted(dic.items(), reverse=True))

        losserbracket = dict(sorted(dic.items()))  # used to select looser values

        count = 0

        for key, values in dic.items():

            self.temp[key] = values

            winners.append(values)
            count += 1

            if (
                count == 20
            ):  # this can variate but we need to keep the time to run the whole efs

                break
        l_count = 0
        for key, values in losserbracket.items():

            winners.append(values)
            l_count += 1

            if l_count == 5:

                break

        self.top_creatures[final] = feat

        print(
            final,
            "generationWinner--------------------------------------------------------------------------------------------------------",
            feat,
            "\n ",
        )

        self.graph.append(final)

        return winners

    def select_features(
        self, variables: list, evalFunction, command="chaos", make_graph=False
    ):

        """
        efs loop for generating features
        Attributes
        ----------
        variables :List
            user input variables for there evaluation  function
        evalFunction:function
            user inputed function name for evaluating the features
        Returns
        -------
        winners:list
            returns list of features
        winningValue:int
            return the best value for feature set
        innovation_dic:
            return dictionary to track items
        vectorized:
            return vector value for features
        list_of_features:
            return a winning set of features
        """
        winners = []

        singularity = [0] * len(self.features)

        for gen in range(self.start, self.generations):

            print(
                "\033[1m",
                "°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸",
                gen,
                "generation",
                "°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸",
                "\n",
                "\033[0m",
            )
            # get starting time
            start = time.time()

            population = [] + winners

            self.binary_tracker = dict(
                sorted(self.binary_tracker.items(), reverse=True)
            )

            key_list = list(self.binary_tracker.keys())

            # crossBreed
            for i in range(len(winners) // 2):

                # TODO a single to perform crossbreeding operation

                breeding = random.sample(winners, 2)  # randomly selected creatures

                ancestor_breeding = random.sample(
                    self.earlier_winners, 1
                )  # randmoly selected from earlier winners

                ancestor_breedingHigh = self.binary_tracker[
                    key_list[0]
                ]  # winners among the earlier selected winners

                # combo_cross_breeding

                genecode = combo_cross_breeding(
                    breeding[0], ancestor_breedingHigh
                )  # crossbreed them

                self.__dictionary_append(
                    breeding[0], ancestor_breedingHigh, genecode, gen
                )  # append them in to dictionary

                population.append(genecode)  # append them into population

                genecode = cross_breeding(
                    breeding[0], ancestor_breedingHigh
                )  # crossbreed them

                self.__dictionary_append(
                    breeding[0], ancestor_breedingHigh, genecode, gen
                )  # append them in to dictionary

                population.append(genecode)  # append them into population

                genecode = random_cross_breeding(
                    breeding[0], ancestor_breedingHigh
                )  # crossbreed them

                self.__dictionary_append(
                    breeding[0], ancestor_breedingHigh, genecode, gen
                )  # append them in to dictionary

                population.append(genecode)  # append them into population

                genecode = cross_breeding(breeding[0], ancestor_breeding[0])

                self.__dictionary_append(breeding[0], ancestor_breeding[0], genecode, gen)

                population.append(genecode)

                genecode = random_cross_breeding(breeding[0], ancestor_breeding[0])

                self.__dictionary_append(breeding[0], ancestor_breeding[0], genecode, gen)

                population.append(genecode)

                genecode = cross_breeding(breeding[0], breeding[1])

                self.__dictionary_append(breeding[0], breeding[1], genecode, gen)

                population.append(genecode)

                genecode = random_cross_breeding(breeding[0], breeding[1])

                self.__dictionary_append(breeding[0], breeding[1], genecode, gen)

                population.append(genecode)

                mask_parent_key = self.__dictionary_key_finder(
                    self.innovation_dic, breeding[0]
                )

                mask = self.innovation_dic[mask_parent_key]["mask"]
                #changed
                genecode = uniformCrossover(breeding[0], breeding[1], mask)

                self.__dictionary_append_mask(
                    breeding[0], breeding[1], genecode, gen, mask
                )

                population.append(genecode)

                mask_parent_key = self.__dictionary_key_finder(
                    self.innovation_dic, ancestor_breeding[0]
                )

                mask = self.innovation_dic[mask_parent_key]["mask"]

                genecode = uniformCrossover(
                    ancestor_breeding[0], breeding[0], mask
                )

                self.__dictionary_append_mask(
                    ancestor_breeding[0], breeding[0], genecode, gen, mask
                )

                population.append(genecode)

                mask_parent_key = self.__dictionary_key_finder(
                    self.innovation_dic, ancestor_breedingHigh
                )

                mask = self.innovation_dic[mask_parent_key]["mask"]

                genecode = uniformCrossover(
                    ancestor_breedingHigh, breeding[1], mask
                )

                self.__dictionary_append_mask(
                    ancestor_breedingHigh, breeding[1], genecode, gen, mask
                )

                population.append(genecode)

            #           mutate
            if command == "chaos":

                temp = winners.copy()

                maybe = 0
                for c in temp:

                    mask_parent_key = self.__dictionary_key_finder(self.innovation_dic, c)

                    mask = self.innovation_dic[mask_parent_key]["mask"]

                    genecode = uniformCrossover(c, "mutate", mask)

                    self.__dictionary_append_mask(c, "mutate", genecode, gen, mask)

                    population.append(genecode)

            indexes = self.__chaotic_population(
                self.total_creatures, gen + 1, len(self.features), 0, None, True
            )

            for index in indexes:

                population.append(create_vector(index, singularity))

                self.innovation_dic[self.innovation_num] = self.dic_format.copy()

                self.innovation_dic[self.innovation_num][
                    "genecode"
                ] = create_vector(index, singularity)

                self.innovation_dic[self.innovation_num]["gen"] = gen

                nice = self.__create_mask(len(self.features),self.features)

                self.innovation_dic[self.innovation_num]["mask"] = nice

                self.innovation_dic[self.innovation_num]["lambda"] = self.__create_mask(6,self.features)

                self.innovation_num += 1

            print("random creatures created", "\n")

            winners = self.__select_winners_top(population, gen, evalFunction, variables)

            self.earlier_winners += winners

            self.binary_tracker.update(self.temp)

            self.temp = {}

            if gen == 0:
                self.individual_winners += winners

            print("winners", len(winners))

            elapsed_time = time.time() - start

            print(
                "time______________________________________________________________________",
                elapsed_time,
            )
            self.time.append(elapsed_time)

        string = self.name + ".npy"

        print(self.top_creatures)

        # self.make_graph([i for i in range(self.generations)],self.time,"time","timeinseconds")

        # self.make_graph([i for i in range(self.start,self.generations)],self.graph,"genwinners","scores")

        np.save(string, self.top_creatures)
        list_of_features = list(self.top_creatures.values())
        # np.save("ans.npy", self.top_creatures)

        np.save("time.npy", self.time)  # saves times

        np.save("graph.npy", self.graph)  # saves graph

        np.save("earlier_winners.npy", self.earlier_winners)

        np.save("binary_tracker.npy", self.binary_tracker)

        # print(self.time)

        print(
            "efs finished", "\n", self.graph, "generation wise scores", "\n", "winners"
        )

        top_value = dict(sorted(self.top_creatures.items(), reverse=True))

        winning_key = list(top_value.keys())[0]

        features = top_value[winning_key]
        print(features, "features")
        vectorized = tracker(features,self.features)

        return features, winning_key, self.innovation_dic, vectorized, list_of_features