#!/bin/bash

# This script will set up SMOG2 contact input files;

PDBfile=$1

export smog2dir=$HOME/bin/smog-2.2

smog2 -i $PDBfile -t $smog2dir/share/templates/SBM_AA -tCG $smog2dir/share/templates/SBM_calpha+gaussian
#smog2 -i $PDBfile -t $smog2dir/share/templates/SBM_AA -tCG $smog2dir/share/templates/SBM_calpha

