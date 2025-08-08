import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import os

import ptBCPNN2
import Measure
import Utils

# To run through all learning rules in non-modular and modular case, gradually increasing number of train patterns until performance drops below 50%. Number of train patterns is increased logarithmically.

def calcTrnpats(trnpat0 = 100, k = 1.5, maxtrnpat = 2000) :

    trnpats = [trnpat0]
    while True :
        trnpat = k * trnpats[-1]
        if trnpat > maxtrnpat :
            break
        trnpats.append(trnpat)

    trnpats = map(int, trnpats)

    return list(trnpats)
        

def runOneLrule(N, K, niter, lrule = "hebb", mode = "run", HDoff = True) :

    trnpats = calcTrnpats(100, 1.25, 5000)

    fcorrs = []
    for trnpat in trnpats :

        if mode == "run" :
            fcorr, train_patterns, test_patterns, recalled, am1 = ptBCPNN2.run(N, K, trnpat, niter = 1,
                                                                               lrule = lrule, HDoff = HDoff)
        else :
            fcorr, train_patterns, test_patterns, recalled, am1 = ptBCPNN2.hrun(N, K, trnpat, niter = 1,
                                                                                lrule = lrule, HDoff = HDoff)
                
        fcorrs.append(fcorr)

        if fcorr < 50 :
            break

    return fcorrs
        

def runbom(paramfile = "BOMmain.par", N = 256, HorK = 16, P = 256, netform = "HxM", gND = 0, niter = 1, nflip = 1,
           kappa = None, seed = 4711, verbosity = 0) :
    modparfile = paramfile + "x"
    if kappa is None :
        kappa = nflip/HorK
        lmbda = 1 - kappa
    else :
        kappa = kappa
        lmbda = kappa
    Utils.modparamvals(parfile = paramfile, modparfile = modparfile,
                       parmods = [["N", N], ["HorK", HorK], ["ntrpat", P], ["netform", netform],
                                  ["hselfex", gND],
                                  ["seed", seed], ["niter", niter],
                                  ["lambda", lmbda], ["kappa", kappa], ["ktehflip", nflip * 1/HorK],
                                  ["verbosity", verbosity]])
    os.system("./BOMmain " + modparfile)
    trinsts0 = Utils.loadbin("trinsts0.bin", N)
    teinsts_dist = Utils.loadbin("teinsts_dist.bin", N)
    recalled = np.loadtxt("recalled.txt")

    fmatchmeanstd = np.loadtxt("BOMfmatch.txt")

    return fmatchmeanstd[0], trinsts0, teinsts_dist, recalled
    

def fcorr_bps_vs_P(N = 400, HorK = 20, nflip = 1, pnflip = 0.1, niter = 15, Pmin = 20, Pmax = 1500, kP = 1.05,
                   lrules = ["hebb", "covar", "precovar", "hopfield", "willshaw", "bcpnn", "boms", "bom"],
                   netform = "HxM", gND = 0, HDoff = True, nrep = 20, measure = 2, verbosity = 1, label = "junk") :

    if nflip is None :
        nflip = pnflip * HorK
        if nflip < 1 :
            raise AssertionError("Illegal: nflip < 1")

    M = N // HorK

    seed = 547208

    for lrule in lrules :

        fext = str(HorK) + label

        filename = lrule + "_" + fext + ".rep"

        open(filename, 'w').close() # Remove old data

        with open(filename, "a") as f:
            Pss = []
            fcorrss = []
            bpsss = []
            
            for rep in range(nrep) :
                Ps = []
                fcorrs = []
                fcorr = 0.11
                bpss = []
                P = Pmin
                while P < Pmax and fcorr > 0.1 :
                    Ps.append(P)

                    if lrule == "bom" :
                        fcorr, train_patterns, test_patterns, recalled = runbom(N = N, HorK = HorK, P = P, netform = netform,
                                                                                niter = niter, nflip = nflip,
                                                                                # kappa = None,
                                                                                seed = seed, verbosity = verbosity)
                        train_patterns = torch.from_numpy(train_patterns)
                        test_patterns = torch.from_numpy(test_patterns)
                        recalled = torch.from_numpy(recalled)
                    else :
                        if netform == "HxM" :
                            fcorr, train_patterns, test_patterns, recalled, net = \
                                ptBCPNN2.hrun(N, HorK, P, nflip = nflip, lrule = lrule, gND = gND,
                                              HDoff = HDoff, niter = niter, seed  = seed, verbosity = verbosity)
                        elif netform == "KofN" :
                            fcorr, train_patterns, test_patterns, recalled, net = \
                                ptBCPNN2.run(N, HorK, P, nflip = nflip, lrule = lrule, gND = gND, niter = niter,
                                             seed  = seed, verbosity = verbosity)
                            HDoff = False
                        else :
                            raise AssertionError("No such netform")
                    fcorrs.append(fcorr)
                    symmetric = lrule != "bom"
                    if measure == 2 :
                        results = Measure.bits_per_synapse2(train_patterns, recalled, HorK, M, gND, HDoff, symmetric)
                    elif measure == 3 :
                        results = Measure.bits_per_synapse3(train_patterns, test_patterns, recalled, HorK, M, HDoff,
                                                            symmetric)
                    bpss.append(results['Bits per Synapse'])

                    print(f"lrule = {lrule} rep = {rep} P = {P} fcorr = {fcorr}% bps = {results['Bits per Synapse']}")

                    P = int(kP * P)
                    
                    seed += 17

                Pss.append(Ps)
                fcorrss.append(fcorrs)
                bpsss.append(bpss)

            Pss = np.array(Pss)
            mean_Ps = np.mean(Pss, 0)
            std_Ps = np.std(Pss, 0)
            fcorrss = np.array(fcorrss)
            mean_fcorrs = np.mean(fcorrss, 0)
            std_fcorrs = np.std(fcorrss, 0)
            bpsss = np.array(bpsss)
            mean_bpss = np.mean(bpsss, 0)
            std_bpss = np.std(bpsss, 0)
            np.savetxt(filename, (mean_Ps, std_Ps, mean_fcorrs, std_fcorrs, mean_bpss, std_bpss))


def plotdata1(HorK = 20, lrules = ["hebb", "covar", "precovar", "hopfield", "willshaw", "bcpnn", "boms", "bom"],
              xlims = (0, 105), ylims = (0, 1), figno = 1, clr = True, label = "_junk") :

    plt.figure(figno)
    if clr : plt.clf()

    colors = ['c', 'g', 'orange', 'b', 'm', 'r', 'lightblue', 'lime']

    for l in range(len(lrules)) :
        lrule = lrules[l] 

        fext = str(HorK) + label

        filename = lrule + "_" + fext + ".rep"

        print("filename =", filename)

        # data = np.loadtxt("resultsSynCap/" + filename)
        data = np.loadtxt(filename)

        ls = '-'

        ax1 = plt.subplot(2,2,1)
        ax1.errorbar(data[0],data[2], data[3], color = colors[l], ls = ls, capsize = 3, label = lrule)
        ax1.plot([0, data[0][-1]],[90, 90],':', color = 'gray')
        ax1.legend(fontsize = 10)
        ax1.set_xlabel("P") ; ax1.set_ylabel("fcorr")
        ax1.set_title("Fraction correct vs patterns stored", fontsize = 12)
        ax2 = plt.subplot(2,2,2)
        ax2.errorbar(data[2],data[4], data[5], color = colors[l], ls = ls, capsize = 3, label = lrule)
        ax2.plot([90, 90], [0, 1.], ':', color = 'gray')

        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        ax2.set_xlabel("fcorr") ; ax2.set_ylabel("C")
        ax2.set_title("Synapse information vs Fraction correct", fontsize = 12)
        ax3 = plt.subplot(2,2,3)
        ax3.errorbar(data[0],data[4], data[5], color = colors[l], ls = ls, capsize = 3, label = lrule)
        ax3.set_xlabel("P") ; ax3.set_ylabel("C")
        ax3.set_title("Synapse information vs patterns stored", fontsize = 12)
        plt.suptitle("Information per synapse (" + fext + ")", fontsize = 20)
        plt.tight_layout()
