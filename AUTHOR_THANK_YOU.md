# Thank you to the paper authors

Thank you, Minhak Song, Liang Zhang, Bingcong Li, Niao He, Michael
Muehlebach, and Sewoong Oh, for developing and sharing *Zeroth-Order
Optimization at the Edge of Stability*.

The paper makes a useful distinction between first-order and zeroth-order
stability: the mean-square ZO boundary depends on the full Hessian spectrum,
while practical tracking can use trace and top-eigenvalue bounds. The explicit
operator formulas, the frozen-preconditioner analysis, and the Section-5
experiments made it possible to audit the theory and the empirical claims as
separate production paths.

Thank you also for making the paper, assumptions, theorem statements, and
experimental setup available. That openness made it possible to reproduce the
theory independently, preserve the CPU-scale limitations honestly, and state
what still needs GPU-scale compute.

This repository is an independent reproduction and audit by
MachineLearning-Nerd. It is not affiliated with, reviewed by, or endorsed by
the authors. Mistakes in the code or interpretation belong to this repository;
corrections are welcome through the normal GitHub channels.
