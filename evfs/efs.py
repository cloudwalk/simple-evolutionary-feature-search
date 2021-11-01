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

    def chaotic_population(
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

    def make_graph(self, x, y, title, yname):

        plt.plot(x, y)

        plt.xlabel("gen")

        plt.ylabel(yname)

        plt.title(title)

        plt.show()

    def dec_to_bin(self, num):

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

    def chaos_equation(self, Lambda, xn):

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

    def mask_converter(self, gray_mask: list, lambda_mask: list):
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

        binary_mask, binary_lambda_mask = self.gray_to_binary(
            gray_mask
        ), self.gray_to_binary(lambda_mask)

        if len(binary_mask) < len(self.features):

            """
            a little trick used to make the length of masks as same as features in future we need to change it

            """

            binary_mask += [0] * len(self.features) - len(binary_mask)  # error chance

        decimal, decimal_lambda = int(binary_mask, 2), int(binary_lambda_mask, 2)

        mask_interval, lambd = self.bin_to_real(
            binary_mask, 0, 1, decimal
        ), self.bin_to_real(binary_lambda_mask, 0, 4, decimal_lambda)

        chaos = self.chaos_equation(round(lambd, 4), round(mask_interval, 4))

        modified_mask_dec = self.masktodecimal(1, 0, chaos, len(self.features))

        modified_mask_dec = round(modified_mask_dec, 4)

        mod_dec_mask = self.dec_to_bin(modified_mask_dec)

        return self.binary_to_gray(mod_dec_mask)

    def xor_conv(self, x1, x2):

        if x1 == x2:
            return "0"
        return "1"

    # Helper function to flip the bit
    def flip(self, c):

        return "1" if (c == "0") else "0"

    def binary_to_gray(self, binary):
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

            gray += self.xor_conv(binary[i - 1], binary[i])

        exten = [0] * (len(self.features) - len(gray))

        exten = "".join(map(str, exten))

        return gray + exten

    def gray_to_binary(self, gray):

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
                binary += self.flip(binary[i - 1])

        exten = [0] * (len(self.features) - len(binary))

        exten = "".join(map(str, exten))

        return binary + exten

    def create_vector(self, indexes, singularity):

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

    def tracker(self, selected_features):

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

            indexes.append(self.features.index(f))

        vectorized = self.create_vector(indexes, [0] * len(self.features))

        return vectorized

    def bin_to_real(self, bits, lower_limit, upper_limit, num):

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

    def masktodecimal(self, upper_limit, lower_limit, num, k):

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

    def uniformCrossover(self, parent1: list, parent2: list, mask: list):

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

    def cross_breeding(self, parent1: list, parent2: list):

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

    def combo_cross_breeding(self, parent1: list, parent2: list):

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

    def random_cross_breeding(self, parent1: list, parent2: list):

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

    def mutate(self, population: list):

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

    def mutate_winners(self, population: list):

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

    def dictionary_key_finder(self, dictionary, genecode):

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

    def dictionary_append(self, parent1_value, parent2_value, genecode, gen):

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
        ] = self.dictionary_key_finder(self.innovation_dic, parent1_value)

        if parent2_value != "mutated":

            self.innovation_dic[self.innovation_num][
                "parent2"
            ] = self.dictionary_key_finder(self.innovation_dic, parent2_value)

        self.innovation_dic[self.innovation_num]["genecode"] = genecode

        self.innovation_dic[self.innovation_num]["gen"] = gen

        self.innovation_dic[self.innovation_num]["mask"] = self.innovation_dic[
            self.innovation_dic[self.innovation_num]["parent1"]
        ]["mask"]

        self.innovation_dic[self.innovation_num]["lambda"] = self.innovation_dic[
            self.innovation_dic[self.innovation_num]["parent1"]
        ]["mask"]

        self.innovation_num += 1

    def dictionary_append_mask(self, parent1_value, parent2_value, genecode, gen, mask):

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

        temp = self.dictionary_key_finder(self.innovation_dic, parent1_value)

        self.innovation_dic[self.innovation_num]["parent1"] = temp

        if parent2_value != "mutated":

            self.innovation_dic[self.innovation_num][
                "parent2"
            ] = self.dictionary_key_finder(self.innovation_dic, parent2_value)
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

        new_mask = self.mask_converter(gray_mask, lambda_mask)

        self.innovation_dic[self.innovation_num]["mask"] = new_mask

        self.innovation_dic[self.innovation_num]["lambda"] = new_lambda

        self.innovation_num += 1

    def dictionary_append_mutate(self, parent, gen):

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
        ] = self.dictionary_key_finder(self.innovation_dic, parent)

        self.innovation_dic[self.innovation_num]["parent2"] = "mutated"

        genecode = self.mutate(parent.copy())

        self.innovation_dic[self.innovation_num]["genecode"] = genecode

        self.innovation_dic[self.innovation_num]["gen"] = gen

        self.innovation_num += 1

        return parent

    def create_mask(self, length):

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
        return self.binary_to_gray(mask)

    def find_input(self, binary_input: list):

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
                action.append(self.features[index])
        return action

    def select_winners_top(self, array: list, gen, evalFunction, variables):

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

            input = self.find_input(creature)

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

                genecode = self.combo_cross_breeding(
                    breeding[0], ancestor_breedingHigh
                )  # crossbreed them

                self.dictionary_append(
                    breeding[0], ancestor_breedingHigh, genecode, gen
                )  # append them in to dictionary

                population.append(genecode)  # append them into population

                genecode = self.cross_breeding(
                    breeding[0], ancestor_breedingHigh
                )  # crossbreed them

                self.dictionary_append(
                    breeding[0], ancestor_breedingHigh, genecode, gen
                )  # append them in to dictionary

                population.append(genecode)  # append them into population

                genecode = self.random_cross_breeding(
                    breeding[0], ancestor_breedingHigh
                )  # crossbreed them

                self.dictionary_append(
                    breeding[0], ancestor_breedingHigh, genecode, gen
                )  # append them in to dictionary

                population.append(genecode)  # append them into population

                genecode = self.cross_breeding(breeding[0], ancestor_breeding[0])

                self.dictionary_append(breeding[0], ancestor_breeding[0], genecode, gen)

                population.append(genecode)

                genecode = self.random_cross_breeding(breeding[0], ancestor_breeding[0])

                self.dictionary_append(breeding[0], ancestor_breeding[0], genecode, gen)

                population.append(genecode)

                genecode = self.cross_breeding(breeding[0], breeding[1])

                self.dictionary_append(breeding[0], breeding[1], genecode, gen)

                population.append(genecode)

                genecode = self.random_cross_breeding(breeding[0], breeding[1])

                self.dictionary_append(breeding[0], breeding[1], genecode, gen)

                population.append(genecode)

                mask_parent_key = self.dictionary_key_finder(
                    self.innovation_dic, breeding[0]
                )

                mask = self.innovation_dic[mask_parent_key]["mask"]

                genecode = self.uniformCrossover(breeding[0], breeding[1], mask)

                self.dictionary_append_mask(
                    breeding[0], breeding[1], genecode, gen, mask
                )

                population.append(genecode)

                mask_parent_key = self.dictionary_key_finder(
                    self.innovation_dic, ancestor_breeding[0]
                )

                mask = self.innovation_dic[mask_parent_key]["mask"]

                genecode = self.uniformCrossover(
                    ancestor_breeding[0], breeding[0], mask
                )

                self.dictionary_append_mask(
                    ancestor_breeding[0], breeding[0], genecode, gen, mask
                )

                population.append(genecode)

                mask_parent_key = self.dictionary_key_finder(
                    self.innovation_dic, ancestor_breedingHigh
                )

                mask = self.innovation_dic[mask_parent_key]["mask"]

                genecode = self.uniformCrossover(
                    ancestor_breedingHigh, breeding[1], mask
                )

                self.dictionary_append_mask(
                    ancestor_breedingHigh, breeding[1], genecode, gen, mask
                )

                population.append(genecode)

            #           mutate
            if command == "chaos":

                temp = winners.copy()

                maybe = 0
                for c in temp:

                    mask_parent_key = self.dictionary_key_finder(self.innovation_dic, c)

                    mask = self.innovation_dic[mask_parent_key]["mask"]

                    genecode = self.uniformCrossover(c, "mutate", mask)

                    self.dictionary_append_mask(c, "mutate", genecode, gen, mask)

                    population.append(genecode)

            indexes = self.chaotic_population(
                self.total_creatures, gen + 1, len(self.features), 0, None, True
            )

            for index in indexes:

                population.append(self.create_vector(index, singularity))

                self.innovation_dic[self.innovation_num] = self.dic_format.copy()

                self.innovation_dic[self.innovation_num][
                    "genecode"
                ] = self.create_vector(index, singularity)

                self.innovation_dic[self.innovation_num]["gen"] = gen

                nice = self.create_mask(len(self.features))

                self.innovation_dic[self.innovation_num]["mask"] = nice

                self.innovation_dic[self.innovation_num]["lambda"] = self.create_mask(6)

                self.innovation_num += 1

            print("random creatures created", "\n")

            winners = self.select_winners_top(population, gen, evalFunction, variables)

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
        vectorized = self.tracker(features)

        return features, winning_key, self.innovation_dic, vectorized, list_of_features
