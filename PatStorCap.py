import numpy as np
import torch
import time

import ptBCPNN2

def hrunsc(lrule, N, H, P, nflip, seed) :
    fcorr, train_patterns, test_patterns, recalled, net = \
        ptBCPNN2.hrun(N, H, P, niter = 15, nflip = 1, lrule = lrule, gND = 0, HDoff = True, seed = seed,
                      verbosity = 0)
    return fcorr


def findx(lrule, N, H, P0, nflip = 1, ntest = 20, nrep = 10, ytrg = 90, seed = 4711, verbosity = 1) :

    start = time.time()

    if verbosity > 0 :
        print("lrule = {:s} N = {:4d} H = {:2d}".format(lrule, N, H), flush = True)

    Pss = []
    fcorrss = []

    for rep in range(nrep) :

        meanPs = []
        stdPs = []
        fcorrs = []    

        P = int(P0 + 0.25 * (np.random.randint(P0) - P0/2))
        Ps = []
        fcorrs = [0]

        knpat = 0.5
        xi = -1

        for i in range(ntest) :

            fcorr = hrunsc(lrule, N, H, P, nflip, seed)

            if verbosity > 1 :
                print("   N = {:4d} H = {:2d} P = {:4.0f} fcorr = {:5.1f}%".
                      format(N, H, P, fcorr), flush = True)

            Ps.append(P)
            fcorrs.append(fcorr)

            if fcorrs[-1] <= ytrg :
                if fcorrs[-2] > ytrg :
                    knpat /= 2
                    if xi < 1 : xi = i
                P /= 1 + knpat
            elif fcorrs[-1] > ytrg :
                if fcorrs[-2] < ytrg :
                    knpat /= 2
                    if xi < 1 : xi = i
                P *= 1 + knpat

            P = int(P)

            if P > 10000 : print("ERROR in findx(), npat > 10000") ; break

            if verbosity > 2 :
                print("i: {:2d} xi = {:2d} knpat = {:.4f}".format(i, xi, knpat))

            if (knpat<=1/P) : break

        if xi < 0 : return 0, 0

        negoffs = -(len(fcorrs) - xi)//2

        meanPs = np.mean(Ps[negoffs:])
        stdPs = np.std(Ps[negoffs:])
        meanfcorrs = np.mean(fcorrs[negoffs:])

        if verbosity > 0 :
            print("  meanPs = {:.2f} stdPs = {:.2f} meanfcorrs = {:5.1f}%". \
                  format(meanPs, stdPs, meanfcorrs), flush = True)

        Pss += Ps[negoffs:]
        fcorrss += fcorrs[negoffs:]

        seed += 17

    print(f"Time elapsed = {time.time() - start} sec")

    return np.mean(np.array(Pss)), np.std(np.array(Pss)), np.mean(np.array(fcorrss))

