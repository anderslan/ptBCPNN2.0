Anders Lansner 20250710

       README file for prBCPNN
       =======================

       This repository contains PyTorch code for a basic recurrent
       associative memory in ptBCPNN.py. It uses simple outer product
       based learning and WTA as activation funtion.

       Non-modular and modular (hypercolumnar) architectures are
       supported.

       A pattern generator exists for creating non-modular and
       hypercolumnar training patterns, and as well as distorted test
       patterns with some units in the training patterns flipped.

       Three different learning rules are implemented, "hebb",
       "willshaw", "bcp", and some more to come.

       A function for measuring bits-per-synapse is included in
       Measure.py.

       Two main functions are included, one to run non-modular and one
       for hypercolumnar networks.

       Getting started
       ---------------

       Start a python terminal and import ptBCPNN and Measure.

       In terminal do:

       returns = ptBCPNNc.hrun(1024, 32, 1775, 4, lrule = "bcpnn", niter = 15, seed = 4711)
       
       Should give output:

       Fraction correct =  90.37 %
       Time elapsed = 1761.2 ms
