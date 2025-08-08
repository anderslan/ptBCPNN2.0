import numpy as np
import torch
from matplotlib import pyplot as plt
import time

import Utils

class PatGen :

    train_patterns = None
    test_patterns = None
    train_instances = None
    test_instances = None


    def __init__(self, seed) :

        self.N = 0
        self.K = 0
        self.H = 0
        self.M = 0
        self.setseed(seed)


    def setseed(self, seed) :
        if seed == 0 :
            seed = int(time.time())
        torch.manual_seed(seed)


    def random_integer_mean(self, target_mean, N, margin = 0.1):
        """
        Generate N integers (all floor or ceil of target_mean) such that their mean is exactly target_mean.

        Args:
            target_mean (float): Desired average
            N (int): Number of integers to generate

        Returns:
            torch.Tensor: [N] tensor of integers, mean = target_mean
        """
        low = int(torch.floor(torch.tensor(target_mean)).item())
        high = low + 1

        total_sum = target_mean * N
        num_high = int(round(total_sum - low * N))  # number of high values (e.g., 4s)
        num_low = N - num_high                      # number of low values (e.g., 3s)

        values = torch.tensor([low] * num_low + [high] * num_high)
        permuted = values[torch.randperm(N)]  # shuffle randomly

        assert abs(permuted.float().mean().item() - target_mean) < margin, "Mean mismatch"

        return permuted


    def generate_K_sparse_binary_patterns(self, N, K, P):
        """
        Generates P binary patterns of length N with exactly K active units per pattern.

        Args:
            P (int): Number of patterns
            N (int): Length of each pattern
            K (int): Number of units to be active (e.g. 0.1)

        Returns:
            torch.Tensor: Tensor of shape [P, N] with binary 0/1 patterns
        """
        self.N = N
        self.K = K
        assert 0 < self.K < self.N, "K must be between 0 and N"

        patterns = torch.zeros((P, self.N))
        for i in range(P):
            active_indices = torch.randperm(self.N)[:self.K]
            patterns[i, active_indices] = 1.0

        self.train_patterns = patterns

        self.test_patterns = None

        return self.train_patterns


    def generate_orthogonal_K_patterns(self, N, K):
        """
        Generate K orthogonal sparse binary patterns of length N.
        Each pattern has exactly one active unit per module (K total),
        and no overlap between patterns (orthogonal).

        Args:
            N (int): Total number of units (must be divisible by K)
            K (int): Number of orthogonal patterns (must be ≤ N / K)

        Returns:
            torch.Tensor: Tensor of shape [K, N] with 0/1 entries
        """
        self.N = N
        self.K = K
        if self.N % self.K != 0 :
            raise ValueError("N must be divisible by K")

        module_size = self.N // self.K

        patterns = torch.zeros((self.K, self.N))

        for p in range(self.K):
            for m in range(self.K):
                global_index = m * module_size + p  # unique index in module m for pattern p
                patterns[p, global_index] = 1.0

        self.train_patterns = patterns

        self.test_patterns = None

        return self.train_patterns


    def  generate_K_distorted_patterns(self, patterns, nflip) :
        """
        Distorts patterns by flipping `nflip` ones to zeros and
        `nflip` zeros to ones in each pattern, preserving total number of 1s.
        """

        distorted_patterns = torch.empty_like(patterns)

        nflips = self.random_integer_mean(nflip, len(patterns))

        for p in range(len(patterns)):
            pattern = patterns[p]
            ones = (pattern == 1).nonzero(as_tuple=True)[0]
            zeros = (pattern == 0).nonzero(as_tuple=True)[0]

            if nflip > len(ones) or nflip > len(zeros):
                raise ValueError(f"Pattern {p}: not enough 1s or 0s to flip {nflip} bits.")

            ones_to_flip = ones[torch.randperm(len(ones))[:nflips[p]]]
            zeros_to_flip = zeros[torch.randperm(len(zeros))[:nflips[p]]]

            distorted = pattern.clone()
            distorted[ones_to_flip] = 0
            distorted[zeros_to_flip] = 1

            distorted_patterns[p] = distorted

        return distorted_patterns


    def generate_K_test_patterns(self, nflip) :
        self.test_patterns = self.generate_K_distorted_patterns(self.train_patterns, nflip)
        return self.test_patterns


    def generate_K_train_instances(self, nflip, ninst) :
        """
        Generates ninst copies of each train_pattern and distorts it with nflip flips.
        """
        train_pattern_copies = np.repeat(self.train_patterns, repeats=ninst, axis=0)

        self.train_instances = self.generate_K_distorted_patterns(train_pattern_copies, nflip)

        return self.train_instances


    def generate_H_sparse_binary_patterns(self, N, H, P):
        """
        Generate P binary patterns over H hypercolumns with M units each.
        Each pattern has one active unit per hypercolumn (one-hot within module).

        Args:
            H (int): Number of hypercolumns (modules)
            M (int): Number of units per hypercolumn
            P (int): Number of patterns to generate

        Returns:
            torch.Tensor: [P, H*M] binary tensor with one 1 per hypercolumn per pattern
        """

        if N % H != 0 :
            raise ValueError("N must be divisible by H")

        self.N = N
        self.H = H
        self.M = N // H

        patterns = torch.zeros((P, self.N))

        for p in range(P):
            for h in range(self.H):
                active_idx = torch.randint(0, self.M, (1,)).item()
                global_idx = h * self.M + active_idx
                patterns[p, global_idx] = 1.0

        self.train_patterns = patterns

        # if filename is not None :
        #     self.train_patterns.numpy().astype('float32').tofile(filename)

        return self.train_patterns


    def generate_orthogonal_H_patterns(self, N, H):
        """
        Generate M orthogonal sparse binary patterns of length N.
        Each pattern has exactly one active unit per hypercolumn (H total),
        and no overlap between patterns (orthogonal).
        """

        if N % H != 0 :
            raise ValueError("N must be divisible by H")

        self.N = N
        self.H = H
        self.M = self.N // self.H

        patterns = torch.zeros((self.M, self.N))

        for p in range(self.M):
            for h in range(self.H):
                global_index = h * self.M + p  # unique index in module m for pattern p
                patterns[p, global_index] = 1.0

        self.train_patterns = patterns
        
        return self.train_patterns


    def generate_H_distorted_patterns(self, patterns, nflip):
        """
        Distort all modular one-hot training patterns by flipping `nflip` hypercolumns per pattern.
        """

        P, N = patterns.shape

        distorted = patterns.clone()

        nflips = self.random_integer_mean(nflip, P)

        for p in range(P):
            pattern = distorted[p]
            flip_indices = torch.randperm(self.H)[:nflips[p]]

            for h in flip_indices:
                start = h * self.M
                end = (h + 1) * self.M
                current_active = pattern[start:end].argmax().item()
                available_indices = list(set(range(self.M)) - {current_active})
                new_idx = available_indices[torch.randint(len(available_indices), (1,)).item()]

                pattern[start:end] = 0
                pattern[start + new_idx] = 1
        
        return distorted


    def generate_H_test_patterns(self, nflip) :
        self.test_patterns = self.generate_H_distorted_patterns(self.train_patterns, nflip)
        return self.test_patterns


    def generate_H_train_instances(self, ninst, nflip) :
        """
        Generates ninst copies of each train_pattern and distorts it with nflip flips.
        """
        self.train_templates = np.repeat(self.train_patterns, repeats=ninst, axis=0)

        self.train_instances = self.generate_H_distorted_patterns(self.train_templates, nflip)

        return self.train_instances


    def generate_H_test_instances(self, ninst, nflip) :
        """
        Generates ninst copies of each train_pattern and distorts it with nflip flips.
        """
        instances = np.repeat(self.train_patterns, repeats=ninst, axis=0)

        self.test_instances = self.generate_H_distorted_patterns(instances, nflip)

        return self.test_instances
        


