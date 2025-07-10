import numpy as np
import torch
from matplotlib import pyplot
import time

import Utils

def setseed(seed) :
    if seed == 0 :
        seed = int(time.time())
    torch.manual_seed(seed)


class PatternGenerator :

    patterns = None
    distorted_patterns = None

    def __init__(self) :

        self.N = 0
        self.K = 0
        self.H = 0
        self.M = 0

    def random_integer_mean(self, target_mean, N):
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

        assert abs(permuted.float().mean().item() - target_mean) < 0.01, "Mean mismatch"

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

        self.patterns = patterns

        self.distorted_patterns = None

        return self.patterns


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

        self.patterns = patterns

        self.distorted_patterns = None

        return self.patterns


    def  generate_K_distorted_patterns(self, nflip):
        """
        Distorts self.patterns by flipping `nflip` ones to zeros and
        `nflip` zeros to ones in each pattern, preserving total number of 1s.
        """

        if self.patterns is None :
            raise AssertionError("No patterns")

        self.distorted_patterns = torch.empty_like(self.patterns)

        P = len(self.patterns)

        nflips = self.random_integer_mean(nflip, P)

        for i in range(P):
            pattern = self.patterns[i]
            ones = (pattern == 1).nonzero(as_tuple=True)[0]
            zeros = (pattern == 0).nonzero(as_tuple=True)[0]

            if nflip > len(ones) or nflip > len(zeros):
                raise ValueError(f"Pattern {i}: not enough 1s or 0s to flip {nflip} bits.")

            ones_to_flip = ones[torch.randperm(len(ones))[:nflips[i]]]
            zeros_to_flip = zeros[torch.randperm(len(zeros))[:nflips[i]]]

            distorted = pattern.clone()
            distorted[ones_to_flip] = 0
            distorted[zeros_to_flip] = 1

            self.distorted_patterns[i] = distorted

        return self.distorted_patterns


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

        self.patterns = patterns

        # if filename is not None :
        #     self.train_patterns.numpy().astype('float32').tofile(filename)

        return self.patterns


    def generate_H_orthogonal_patterns(self, N, H):
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

        self.patterns = patterns
        
        return self.patterns


    def generate_H_distorted_patterns(self, nflip):
        """
        Distort all modular one-hot training patterns by flipping `nflip` hypercolumns per pattern.
        """

        P, N = self.patterns.shape

        self.distorted_patterns = self.patterns.clone()

        nflips = self.random_integer_mean(nflip, P)

        for p in range(P):
            pattern = self.distorted_patterns[p]
            flip_indices = torch.randperm(self.H)[:nflips[p]]

            for h in flip_indices:
                start = h * self.M
                end = (h + 1) * self.M

                current_active = pattern[start:end].argmax().item()
                available_indices = list(set(range(self.M)) - {current_active})
                new_idx = available_indices[torch.randint(len(available_indices), (1,)).item()]

                pattern[start:end] = 0
                pattern[start + new_idx] = 1
        
        return self.distorted_patterns


class AM :

    def __init__(self, N, K, lrule = "willshaw", HDoff = True) :
        self.N = N
        self.K = K
        self.lrule = lrule
        self.HDoff = HDoff;
        self.Cij = None
        self.Ci = None
        self.C = 0
        self.Pij = None
        self.Pi = None
        self.Wij = None
        self.Bj = None
        self.train_patterns = None
        self.test_patterns = None


    def compute_Cij_Ci_C(self) :
        """
        Compute both outer product sum and per-unit activation sum from binary train_patterns.

        Args:
            train_patterns (torch.Tensor): [P, N] tensor with binary train_patterns (0/1)

        Returns:
            Cij (torch.Tensor): [N, N] sum of outer products
            activations (torch.Tensor): [N] sum of activations per unit
        """
        P = len(self.train_patterns)
        self.Cij = torch.zeros((self.N, self.N))

        for mu in range(P):
            pattern = self.train_patterns[mu].unsqueeze(1)  # shape [N, 1]
            self.Cij += pattern @ pattern.T           # outer product

        self.Ci = self.train_patterns.sum(dim=0)  # sum over train_patterns → shape [N]

        self.C = len(self.train_patterns)


    def willshaw_w(self):
        """
        Apply Willshaw learning rule: W[i,j] = 1 if C[i,j] > 0, else 0.

        Args:
            C (torch.Tensor): Correlation matrix (outer product sum), shape [N, N]

        Returns:
            torch.Tensor: Binary weight matrix W, same shape as C
        """

        self.Bj = torch.zeros(self.N)

        eps = 1e-6

        self.Pij = (self.Cij / self.C).clamp(min=eps*eps)

        self.Wij = (self.Pij > eps*eps).float()

        return self.Wij


    def hebbian_w(self):
        """
        Apply Hebbian learning rule: W[i,j] = C[i,j]

        Args:
            C (torch.Tensor): Correlation matrix (outer product sum), shape [N, N]

        Returns:
            torch.Tensor: Binary weight matrix W, same shape as C
        """

        print("hebbian")

        self.Bj = torch.zeros(self.N)

        eps = 1e-6

        self.Pij = (self.Cij / self.C).clamp(min=eps*eps)

        self.Wij = self.Pij

        return self.Wij


    def bcpnn_w(self):
        """
        Compute BCPNN log-weight matrix from outer product sums and activations.

        Args:
            Cij (torch.Tensor): [N, N] matrix of co-activation counts
            Ci (torch.Tensor): [N] vector of unit activation counts
            C (int): Total number of patterns

        Returns:
            torch.Tensor: [N, N] matrix of BCPNN log-weights
        """
        print("bcpnn")

        # To avoid division by zero or log(0), we clamp values
        eps = 1/(self.C + 1)
        self.Ci = self.Ci.clamp(min=eps)
        # self.Cj == self.Ci

        self.Pi = (self.Ci / self.C).clamp(min=eps)

        self.Pij = (self.Cij / self.C).clamp(min=eps*eps)

        numerator = self.Pij
        denominator = self.Pi.unsqueeze(1) * self.Pi.unsqueeze(0)  # Outer product: [N, N]

        ratio = numerator / denominator

        self.Bj = torch.log(self.Pi)

        self.Wij = torch.log(ratio)

        return self.Wij


    def remove_self_connections(self):
        """
        Zero out the diagonal of an N x N weight matrix to remove self-connections.
        """
        assert self.Wij.shape[0] == self.Wij.shape[1], "Weight matrix must be square"
        self.Wij.fill_diagonal_(0)


    def train(self, train_patterns):

        if train_patterns.shape[1] != self.N :
            raise AssertionError(f"Pattern width -- network N mismatch {train_patterns.shape[1]} vs {self.N}")

        self.train_patterns = train_patterns
        self.compute_Cij_Ci_C()
        if self.lrule == "willshaw" :
            self.Wij = self.willshaw_w()
        elif self.lrule == "hebbian" :
            self.Wij = self.hebbian_w()
        elif self.lrule == "bcpnn" :
            self.Wij = self.bcpnn_w()
        else :
            raise ValueError(f"Unsupported learning rule: {self.lrule}. Choose 'willshaw'/'hebbian'/bcpnn.")
        if self.HDoff :
            self.remove_self_connections()

    def K_wta(self) :
        # Apply K-Winner-Take-All
        # For each row, set top-K indices to 1, rest to 0
        topk_vals, topk_indices = torch.topk(self.supports, self.K, dim=1)

        self.recalled = torch.zeros_like(self.supports)
        for i in range(self.test_patterns.size(0)):
            self.recalled[i, topk_indices[i]] = 1.0


    def recall(self, test_patterns, niter = 1):
        """
        Recall function using distorted train_patterns.
        """
        if niter < 1 :
            raise AssertionError("Illegal: niter < 1")

        self.test_patterns = test_patterns

        input_pattern = self.test_patterns
        # Linear activation: test_patterns * W
        for iter in range(niter) :
            self.supports = self.Bj + input_pattern @ self.Wij  # shape: [B, N]
            self.K_wta()
            input_pattern = self.recalled

        return self.recalled


class HAM(AM) :

    def __init__(self, N, H, lrule = "willshaw", HDoff = True) :
        """
        Hypercolumnar Associative Memory subclass of AM.
        """
        super().__init__(N = N, K = 0, lrule=lrule, HDoff = HDoff)

        self.H = H
        if N % H != 0 :
            raise ValueError("N must be divisible by H")
        self.M = N // H


    def remove_self_connections(self) :
        """
        Zero out the MxM block diagonals in an (H*M) x (H*M) weight matrix.
        """
        for h in range(self.H):
            start = h * self.M
            end = (h + 1) * self.M
            self.Wij[start:end, start:end] = 0


    def H_wta(self):
        """
        Apply 1-Winner-Take-All within each of H hypercolumns (modules of size M).

        Args:
            supports (torch.Tensor): [B, H*M] tensor of real-valued activations
            H (int): Number of hypercolumns
            M (int): Number of units per hypercolumn

        Returns:
            torch.Tensor: [B, H*M] binary tensor with one winner per hypercolumn
        """
        B, N = self.supports.shape

        self.recalled = torch.zeros_like(self.supports)

        for h in range(self.H):
            start = h * self.M
            end = (h + 1) * self.M

            # Get self.supports in this hypercolumn: [B, M]
            block = self.supports[:, start:end]

            # Find winners: [B]
            winners = torch.argmax(block, dim=1)

            # Set winners to 1
            self.recalled[torch.arange(B), start + winners] = 1.0

        return self.recalled


    def recall(self, test_patterns, niter = 1):
        """
        Recall function using distorted train_patterns
        """
        if niter < 1 :
            raise AssertionError("Illegal: niter < 1")

        # Linear activation: test_patterns * W
        input_patterns = test_patterns

        Bj = self.Bj.expand(len(test_patterns), -1)  # efficient, no memory copy
        for iter in range(niter) :
            self.supports = Bj + input_patterns @ self.Wij  # shape: [B, N]
            self.H_wta()
            input_patterns = self.recalled

        return self.recalled

def calc_fcorrect(train_patterns, recalled) :
    P = len(train_patterns)
    K = torch.sum(train_patterns[0])
    ntest = P * K
    nerr = np.sum(np.abs(np.array(recalled - train_patterns))/2)
    print("Fraction correct =  {:.2f} %".format(100 * (ntest - nerr)/ntest))


def calc_fcorrectc(train_patterns, recalled) :
    P = len(train_patterns)
    K = torch.sum(train_patterns[0])
    nerr = np.sum(np.sum(np.abs(np.array(recalled - train_patterns))/2, 1) > 0)
    return 100 * (P - nerr)/P


def run(N, K, P, nflip, niter = 1, lrule = "willshaw", HDoff = True, seed = 0) :
    setseed(seed)
    start = time.time()
    am1 = AM(N, K, lrule, HDoff)
    pg1 = PatternGenerator()
    train_patterns = pg1.generate_K_sparse_binary_patterns(N, K, P)
    am1.train(train_patterns)
    test_patterns = pg1.generate_K_distorted_patterns(nflip)
    recalled = am1.recall(test_patterns, niter)
    fcorr = calc_fcorrectc(am1.train_patterns, am1.recalled)
    print("Fraction correct =  {:.2f} %".format(fcorr))
    print("Time elapsed = {:.1f} ms".format(1000 * (time.time() - start)))
    return train_patterns, test_patterns, recalled, am1


def hrun(N, H, P, nflip, niter = 1, lrule = "willshaw", HDoff = True, seed = 0) :
    setseed(seed)
    start = time.time()
    if N % H != 0 :
        raise AssertionError(f"N%H in nonzero (N = {N}, H = {H})")
    ham1 = HAM(N, H, lrule, HDoff)
    pg1 = PatternGenerator()
    train_patterns = pg1.generate_H_sparse_binary_patterns(N, H, P)
    ham1.train(train_patterns)
    test_patterns = pg1.generate_H_distorted_patterns(nflip)
    recalled = ham1.recall(test_patterns, niter)
    fcorr = calc_fcorrectc(train_patterns, ham1.recalled)
    print("Fraction correct =  {:.2f} %".format(fcorr))
    print("Time elapsed = {:.1f} ms".format(1000 * (time.time() - start)))
    return train_patterns, test_patterns, recalled, ham1
