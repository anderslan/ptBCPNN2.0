Anders Lansner 20250710

       README file for prBCPNN2
       ========================

       This repository contains PyTorch code for a basic recurrent
       associative memory in ptBCPNN.py. It uses simple outer product
       based learning and WTA as activation funtion.

       Non-modular and modular (hypercolumnar) architectures are
       supported.

       A pattern generator exists for creating non-modular and
       hypercolumnar training patterns, and as well as distorted test
       patterns with some units in the training patterns flipped.

       Synchronous and asynchronous update methods are implemented and
       can be tested by commenting/uncommenting in run() and hrun(),

       Seven different learning rules are implemented, "hebb",
       "covar", "precovar", "hopfield", "willshaw", "bcpnn", and
       "boms" have been implemented.

       Functions for measuring bits-per-synapse are included in
       Measure.py.

       Two main functions are included, one to run non-modular and one
       for hypercolumnar networks.

       In First_tests.py and PatStorCap.py are en embryo of tools to
       evaluate performance.

       Getting started
       ---------------

       Start a python terminal and import ptBCPNN2.

       In terminal do:

       returns = ptBCPNN2.hrun(1024, 32, 1775, niter = 15, nflip = 1, lrule = "bcpnn", seed = 4711, verbosity = 3)
       
       Should give about output:

       Fraction correct =  92.90 %
       Time elapsed = 1089.5 ms
