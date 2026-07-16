This directory contains the official CUTEst set of optimization test
problems expressed in Standard Input Format (SIF). Each example is
contained in a file with the extension .SIF (but N.B., below). The 
leading comments describe briefly the source of the example, with 
citations if relevant, and contains a classification string that conforms 
to rules described in

  https://ralna.github.io/SIFDecode/html/classification/

For a formal definition of SIF, see

  https://ralna.github.io/SIFDecode/html/sif/

In addition, for convenience, there are subdirectories that provide symbolic 
links to all the CUTEst examples in commonly-occurring subsets. These are

 lpsif Linear Programming examples 
 qpsif Quadratic Programming examples 
 bqpsif Bound-constrained Quadratic Programming examples 
 nlssif Nonlinear least-squares examples 
 nlesif Nonlinear equation and inequality examples 
 uncsif Unconstrained examples 
 bcsif Bound-Constrained examples 
 gosif Global Optimization examples 
 lcsif Linearly-constrained nonlinear programming examples 
 nlpsif Nonlinear Programming examples

Each directory contains a file *.dat listing the examples in alphabetical order.

Currently, the problems encoded as

  BA-L16LS.SIF, BA-L52LS.SIF, LRCOVTYPE.SIF, MNISTS0.SIF, MNISTS5.SIF
  BA-L16.SIF, BA-L52.SIF, MNISTS0LS.SIF and MNISTS5LS.SIF

exceed the permitted Github filesize limit, and we provide them instead as 
compressed (bzip2) files. Once they have been downloaded, they may be
restored using (in Unix-like systems) the commands

  bunzip2 -k BA-L16LS.SIF.bz2 
  
etc; the -k flag may be ommited if a copy of the original (compressed) file is 
not required.

Further subsets may be found using the SIFDecode select tool from

  https://github.com/ralna/SIFDecode/wiki

Nick Gould, for GALAHAD productions 31st May, 2021
