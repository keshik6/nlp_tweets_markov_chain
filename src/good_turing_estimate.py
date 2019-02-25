# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 00:30:21 2018

@author: Keshik

References: 
    https://github.com/maxbane/simplegoodturing
    https://github.com/georgepar/simple-good-turing
    https://github.com/karanmotani/bigram-probabilities
    https://github.com/zack112358/simple-good-turing-estimation
"""

from __future__ import division
import numpy as np
import collections


# the implementation and the symbols are based on
# [Gale & Sampson, 1995]
class SimpleGoodTuring():
    def __init__(self, species, max_count):
        self._sort_dict = lambda my_dict: \
            collections.OrderedDict(sorted(my_dict.items()))
        self.__species, self.__max_count = species, max_count
        self.__r, self.__n, self.__Z, self.__S, self.__P0, self.__sgt_probs = \
            None, None, None, None, None, None
    
    
    """
    @notice: compute the count of counts for the different individuals in the species
    @param species: dict, initial measurement
    @param max_count: int, most popular element count
    @return: (r, n) = (np.array, np.array):#individuals in species, count of species with r individuals
    """
    def count_of_counts(self, species, max_count):
        N_c = dict()
        for r in range(1, max_count + 1):
            n = np.sum(np.fromiter(
                (1 for v in species.values() if v == r), np.int))
            if n != 0:
                N_c[r] = n
        sort_Nc = self._sort_dict(N_c)
        #print(sort_Nc.keys())
        r = np.array(list(sort_Nc.keys()), dtype=int)
        n = np.array(list(sort_Nc.values()), dtype=int)
        return r, n

    
    """
    @notice: Z smooths the count of counts by "interpolating" each element with the previous and the next value
    @param r: numpy.array, number of individuals of certain species
    @param n: numpy.array, number of different species with r individuals
    @return: np.array, smoothed count of counts
    """
    def Z_smoothing(self, r, n):
        Z = dict()
        if len(r) == 1:
            return np.array([1,1], dtype=float)
        Z[r[0]] = 2 * n[0] / r[1]
        Z[r[-1]] = n[-1] / (r[-1] - r[-2])
        for idx in range(1, len(r) - 1):
            j, nj = r[idx], n[idx]
            i, k = r[idx - 1], r[idx + 1]
            Z[j] = 2 * nj / (k - i)
        sort_Z = self._sort_dict(Z)
        Z_arr = np.array(list(sort_Z.values()), dtype=float)
        return Z_arr
    
    
    """
    @notice: Computes best fit line a+b*log(r) to the pairs log(r) - log(Z)
    @param r: numpy.array
    @param Z: numpy.array, smoothed count of counts
    @return: (a, b, S) = (int, int, numpy.array): cross of y axis, slope of line, antilog(a+b*log(r))
    """
    def linregres_coeffs(self, r, Z):
        if len(r) == 1:
            return 0, 0
        
        logr = np.log10(r)
        logZ = np.log10(Z)
        A = np.vstack([logr, np.ones(len(logr))]).T
        b, a = np.linalg.lstsq(A, logZ)[0]
        return a, b
    
    """
    @notice: Computes Transformation function to fill gaps
    @param a: y intercept
    @param b: slope
    @param r: #individuals in species
    @return: transformed function
    """
    def compute_S(self, a, b, r):
        return 10 ** (a + b * np.log(r))
    
    
    """
    Compute smoothed (discounted) count values r*
    :param r: numpy.array
    :param n: numpy.array
    :param S: numpy.array
    :return: numpy.array, r* - smoothed counts
    """
    def smooth_counts(self, r, n, S):
        r_star = np.array(r, dtype=float)
        cease_x = False
        for i in np.arange(len(r)):
            if (r[i] not in np.arange(len(S))):
                y = y = (r[i] + 1) * S[i] / S[i- 1]
            else:
                y = (r[i] + 1) * S[r[i]] / S[r[i] - 1]  # zero index base in python
                #print("Sr+1 = " + str(S[r[i]]))
                #print("Sr = " + str(S[r[i] - 1]))
            
            x = 0
            t = float("inf")
            if i + 1 >= len(r) or r[i + 1] != r[i] + 1:
                cease_x = True
            if not cease_x:
                x = (r[i] + 1) * n[i + 1] / n[i]
                t = 1.96 * np.sqrt((r[i] + 1) * (r[i] + 1) * n[i + 1] *
                                   (1 + n[i + 1] / n[i]) / (n[i] * n[i]))
            if np.abs(x - y) <= t:
                cease_x = True
                r_star[i] = y
            else:
                r_star[i] = x
        return r_star
    
    
    """
    @notice: Compute P0 and Good-Turing probabilities discount
    @param r: numpy.array
    @param r_star: numpy.array, smoothed r
    @param n: numpy.array
    @param S: numpy.array
    @return: (P0, sgt_probs) = (float, numpy.array): the probability of an unseen species, sgt probabilities discount
    """
    def sgt_discount(self, r, r_star, n):
        P0 = r[0] / np.inner(r, n)
        sgt_probs = (1 - P0) * r_star / np.inner(r_star, n)
        return P0, sgt_probs
    
    
    """
    @notice: Compute smoothed probabilites
    @param species: dict
    @param P0: float
    @param sgt_probs: numpy.array
    @param species_pool: list, all the possible species
    @return: dict, the probability of any seen species if species_pool is specified the probability of the unseen species is also calculated
    """
    def species_probs(self, species, r, P0, sgt_probs, species_pool=[]):
        species_sgt = dict()
        if species_pool == None:
            species_pool = []
        if len(species_pool) == 0:
            species_pool = species.keys()
        species_seen = species.keys()
        species_unseen = list(set(species_seen) ^ set(species_pool))
        for s in species_seen:
            idx = -1
            try:
                idx = np.where(r == species[s])
            except KeyError:
                print("Value not found in r table.\n")
                print("Maybe used different species for computing sgt probs?\n")
            species_sgt[s] = sgt_probs[idx][0]

        total_unseen = len(species_unseen)
        for s in species_unseen:
            species_sgt[s] = P0 / total_unseen
        return species_sgt
    
    
    """
    @notice: Complete wrapper to compute smoothed probabilites
    @param species_pool: list, all the possible species
    @return: dict, the probability of any seen species if species_pool is specified the probability of the unseen species is also calculated
    """
    def run_sgt(self, species_pool=[]):
        self.r, self.n = self.count_of_counts(self.species, self.max_count)
        self.Z = self.Z_smoothing(self.r, self.n)
        a, b = self.linregres_coeffs(self.r, self.Z)
        self.S = self.compute_S(a, b, np.arange(1, self.max_count + 2))
        r_star = self.smooth_counts(self.r, self.n, self.S)
        self.P0, self.sgt_probs = self.sgt_discount(self.r, r_star, self.n)
        species_sgt = self.species_probs(self.species,
                                         self.r, self.P0,
                                         self.sgt_probs,
                                         species_pool)
        return species_sgt

    
    @property
    def species(self):
        return self.__species
    
    
    @property
    def max_count(self):
        return self.__max_count
    
    
    @property
    def r(self):
        return self.__r
    
    
    @r.setter
    def r(self, r):
        self.__r = r
    
    
    @property
    def n(self):
        return self.__n
    
    
    @n.setter
    def n(self, n):
        self.__n = n
    
    
    @property
    def Z(self):
        return self.__Z
    
    
    @Z.setter
    def Z(self, Z):
        self.__Z = Z
    
    
    @property
    def S(self):
        return self.__S
    
    
    @S.setter
    def S(self, S):
        self.__S = S
    
    
    @property
    def P0(self):
        return self.__P0
    
    
    @P0.setter
    def P0(self, P0):
        self.__P0 = P0
    
    
    @property
    def sgt_probs(self):
        return self.__sgt_probs
    
    
    @sgt_probs.setter
    def sgt_probs(self, S):
        self.__sgt_probs = S
