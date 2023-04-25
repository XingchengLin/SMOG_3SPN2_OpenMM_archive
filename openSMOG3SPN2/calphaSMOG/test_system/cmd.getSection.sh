#########################################################################
# Author: Xingcheng Lin
# Created Time: Wed Apr 17 18:40:46 2019
# File Name: cmd.removeidr.sh
# Description: Remove the disordered region of PRC2 in smog.top
#########################################################################
#!/bin/bash

python getSection.py ./smog.top atoms.dat "[ atoms ]" "[ bonds ]"
python getSection.py ./smog.top bonds.dat "[ bonds ]" "[ angles ]"
python getSection.py ./smog.top angles.dat "[ angles ]" "[ dihedrals ]"
python getSection.py ./smog.top dihedrals.dat "[ dihedrals ]" "[ pairs ]"
python getSection.py ./smog.top pairs.dat "[ pairs ]" "[ exclusions ]"
python getSection.py ./smog.top exclusions.dat "[ exclusions ]" "[ system ]"

