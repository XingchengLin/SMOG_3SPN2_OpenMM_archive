# SMOG_3SPN

This is the project of applying SMOG and 3SPN to openmm. This can be used to simulate protein-DNA systems, such as chromatin fibers. Now we can do implicit ions, and we are working on applying explicit ions. 

Proteins are modeled by SMOG2, which is a C-alpha structure-based coarse-grained model. 
SMOG2 paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004794

DNA is modeled by 3SPN or 3SPN.2C, which uses 3 coarse-grained sites for one nucleotide. Originally 3SPN and 3SPN.2C are published as lammps version, and later modified to openmm version. The related publications are: 
3SPN paper: https://aip.scitation.org/doi/full/10.1063/1.4822042
3SPN.2C paper: https://aip.scitation.org/doi/10.1063/1.4897649
Open3SPN2 (openmm version of 3SPN2 and 3SPN.2C) paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008308

Important note for using this to simulation protein-DNA system:
(1) All the nonbonded interactions (van der Waals, electrostatic interactions) have to share the same exclusion list, so usually the exclusion list is very long. For large system, this may lead to large burden for the memory, and one solution is to combine multiple types of nonbonded interactions into one single type, thus the exclusion list does not need to be repeated many times. 
(2) You should decide whether to remove W-C paired nucleotides from nonbonded interactions (related to variable `OpenCLPatch` in `ff3SPN2.py`).

Important note for using this to simulation chromatin system:
(1) The default way to apply rigid body is to rigidize histone core and middle 73bp core DNA. This helps decrease the length of exclusion list, and we can use larger timestep to accelerate. 

Our code has the following mode: 
(1) "default": Apply rigid body, W-C paired bases have excluded volume interactions, protein native pairs have nonbonded interactions. The logic for this mode is to shorten the exclusion list and minimize the system size to save memory (protein native pairs have nonbonded interactions, which does not affect the results since histone core is rigid), and also with rigid body settings we can use large timestep.
(2) "expensive": Not apply rigid body, W-C paired bases do not have excluded volume interactions, protein native pairs do not have nonbonded interactions. This mode is the most expensive to perform, but the most accurate in principle.
(3) "compare1": Not apply rigid body, W-C paired bases do not have excluded volume interactions, protein native pairs have nonbonded interactions. This mode is used to compare with lammps results (note this mode is only for comparison and debug, as in principle if protein native core is not rigid and structure based native pairs are applied, then there should be no additional nonbonded interactions between native pairs). 
(4) "compare2": Not apply rigid body, W-C paired bases have excluded volume interactions, protein native pairs have nonbonded interactions. 

# This is the branch for implementing SMOG-based Calpha SBM with 3SPN.2C
# The code was adapted from the Open3SPN/OpenAWSEM
#
# The implementation is still ongoing
# Stay tuned :)
# Important: Apply MJ potential for protein-protein pair-wise nonbonded interactions

# Some instructions and notes
open3spn2

DNA forcefield parameter file is openSMOG3SPN2/open3SPN2/3SPN2.conf

Some notes about 3SPN2.conf:

`[Particles]`
`name    =   DNA name    epsilon	radius	mass	charge`
DNA: DNA type, can be A or B or B_curved
The excluded volume interactin parameters between any two DNA CG atoms are defined. The excluded volume effect is captured by LJ potential that has a cutoff equal to LJ potential $\sigma$. The radius of each DNA CG atom is given and LJ potential $\sigma$ is the mean of the radiuses of the two atoms involved. LJ potential $\epsilon$ is the geometric mean of the epsilon values of the two atoms involved. Check J. Chem. Phys. 139, 144903 (2013) for details about which CG DNA atom pairs are involved in this excluded volume potential.  

`[Bonds]`
`name = DNA i j s1 r0 Kb2 Kb3 Kb4`
DNA: DNA type, can be A or B or B_curved
i, j: atom type, bond is applied between these two types of atoms
s1: residue index difference
If atom type i is in residue index r, and atom type j is in residue index r + s1, and these two atoms are in the same chain, then the bond is applied. 
Bond potential form: `Kb2*(r-r0)^2+Kb3*(r-r0)^3+Kb4*(r-r0)^4` (this form is defined in `class Bond` in ff3SPN2.py)

`[Harmonic Angles]`
`name	=	DNA	i	j	k	s1	s2	epsilon	t0	Base1	Base2	sB`
DNA: DNA type, can be A or B or B_curved
If 3 atoms (aai, aaj, aak) satisfy all the follwing conditions:
(1) (aai, aaj, aak) are of atom types (i, j, k) 
(2) (aai, aaj, aak) are in the same chain and with residue index (r, r + s1, r + s2)
(3) An atom aax exists, which is of type 'S' in the same chain as aai, and with residue index r + sB. This is used to determine the sequentially +1 or -1 neighboring base type (sB = 1 or -1), since for B_curved DNA type, the angle potential depends on not only the base type for atom aai, but also depends on the base type for the +1 or -1 residue, which is captured by the base type of atom aax (note base type is based on resname, which is DC or DG or DT or DA, so even if atom type is 'S' or 'P', it still has a corresponding base type. In other words, <font color=DeepSkyBlue>here base type is equivalent to residue type</font>). 
(4) Atom aai belongs to base type Base1 and atom aax belongs to base type Base2 (Similar to (3), here the base type is extracted from resname, so atom type 'S' has a corresponding base type).
Note angle t0 is in unit degree, and openmm will convert this from unit degree to unit radian (check `class Angle` in ff3SPN2.py)
Then angle potential will be applied. The angle potential from is: `epsilon*(theta-t0)^2` (this form is defined in `class Angle` in ff3SPN2.py). Note the default setting for openmm HarmonicAngleForce is of form `0.5*k_angle*(theta-theta0)^2`, but here k_angle is defined as 2*epsilon. 
Check J. Chem. Phys. 141, 165103 (2014) for detailed explanations about such sequence dependent angle potential. 

`[Base Stackings]`
`name	=	DNA	i	j	k	s1	s2	epsilon	sigma	t0	alpha	rng`
DNA: DNA type, can be A or B or B_curved
If 3 atoms (aai, aaj, aak) are of atom types (i, j, k), and within the same chain, and with residue index (r, r + s1, r + s2), then a base stacking potential is applied to these 3 atoms. In fact, s1 is always 0 and s2 is always 1. Atom type i is always S, while atom types j, k are bases. So this potential captures stackings between neighboring bases within the same ssDNA (i.e. intra-strand base stacking). 
The stacking potential is a complex 3-body potential with both base-base distance and sugar-base-base (aai is atom type S, while aaj and aak are atom type base) . Check `class Stacking` in ff3SPN2.py for details. Read J. Chem. Phys. 139, 144903 (2013) for more explanations about the intra-strand stacking potential (note in 3SPN.2C, intra-strand base stacking interaction is the same as the one in 3SPN.2). So epsilon is the strength coefficient for the Morse potential, sigma is the distance between two bases at equilibrium, t0 is the sugar-base-base angle in crystal structure, alpha is the parameter for controling the range of Morse potential, and rng is the parameter that controls the range of angle-dependent term (that is K in eqn 4 of paper J. Chem. Phys. 139, 144903 (2013)). 

`[Dihedrals]`
`name	=	DNA	i	j	k	l	s1	s2	s3	dihedral_type	K_dihedral	K_gaussian	sigma	t0`
DNA: DNA type, can be A or B or B_curved
If 4 atoms (aai, aaj, aak, aal) are of atom types (i, j, k, l), and within the same chain, and with residue index (r, r + s1, r + s2, r + s3), then a dihedral potential is applied. The potential form is `K_periodic*(1-cs)-K_gaussian*exp(-dt_periodic^2/2/sigma^2)`, with `cs = cos(dt); dt_periodic = dt-floor((dt+pi)/(2*pi))*(2*pi); dt = theta-t0`. So the dihedral potential is the sum of a periodic term including cos function, and a Gaussian function. For B_curved DNA type, t0 is read from the template structure. 

`[Base Pairs]`
`name	=	DNA	Base1	Base2	sigma	alpha	rang	epsilon	torsion	t1	t2`
DNA: DNA type, can be A or B or B_curved
Base1 and Base2 are base atom types, where Base1 is donor and Base2 is acceptor.
The definitions of parameters sigma, alpha, rang, epsilon, torsion, t1, and t2 are included in ff3SPN2.py `class BasePair` method `defineInteraction`. If my understanding is correct, torsion is the reference dihedral value (i.e. S-base-base-S dihedral); sigma is the equilibrium distance between two CG base atoms; t1 and t2 are reference values for angles d2-d1-a1 and a2-a1-d1, respectively; rang controls range of the angle dependent term; epsilon is Morse potential strength coefficient; and alpha controls the range of Morse potential. 
Basically base pairing is a non-specific interaction, just similar to LJ potential, which should exist for any two atoms of specific types within some distance range. So the donor is the base atom and its sugar atom, and the acceptor is also the base atom and its sugar. Base pair interactions are excluded if two bases are within the same ssDNA and the absolute value of their resSeq difference is no larger than 2 (i.e. `(atom_a.chainID == atom_b.chainID) and (abs(atom_a.resSeq - atom_b.resSeq) <= 2)`). 

`[Cross Stackings]`
`name = DNA Base_d1 Base_a1 Base_a3 t03 T0CS_1 T0CS_2 Sigma_1 Sigma_2 alpha_cs1 alpha_cs2 rng_cs1 rng_cs2 rng_bp eps_cs1 eps_cs2`
DNA: DNA type, can be A or B or B_curved
Note there are two sets of parameters for every `Base_d1`, `Base_a1`, and `Base_a3`, for example, there are `T0CS_1` and `T0CS_2`. This is because every ssDNA is different from 5' to 3'. That is to say, for example, two sequences, 'AG' and 'GA' (both sequences from 5' to 3'), are actually different. Let's take a detailed example: if base atom A1 (A means it is type A, and 1 is the index) now pairs with base atom T3 from sequence C2-T3-C4 (i.e. the sequence is CTC, and base atom indexes are 2, 3, and 4, respectively), then the cross stacking interaction parameters are different between A1-C2 and A1-C4, though they are both cross stackings between A and C. 
$\theta_{CS}$ in J. Chem. Phys. 139, 144903 (2013) is defined as the angle between [donor atom 2 - donor atom 1 - acceptor atom 3], where donor atom 2 is type S, while donor atom 1, acceptor atom 3 are both CG base atoms. 
Similar to base pair interaction, cross stacking is also a non-specific interaction, and as long as donor atom 1 forms W-C pair with acceptor atom 1 (you can see `Base_d1` and `Base_a1` are always in W-C pair), then donor atom 1 and acceptor atom 3 form cross stacking interaction. If donor atom 1 and acceptor atom 1 do not form W-C pair, then there is no such cross stacking interaction. 

The bonds, angles, stackings, and dihedrals are applied with function `computeTopology`, which analyzes the CG DNA atom type, chainID and resID to determine the forcefield. One potential way to accelerate the step for analyzing topology is to save all the bonds, angles, stackings, and dihedrals into a text file and load them without running `computeTopology`. 

For base pairing and cross-stacking interactions, there are two classes for these two forces, respectively. Check the notes about ff3SPN2.py. 


Some notes about ff3SPN2.py:

`class DNA(object)`
Class for DNA

`parseConfigurationFile(self, configuration_file=f'{__location__}/3SPN2.conf')`
Load DNA forcefield parameters saved in 3SPN2.conf

`computeTopology(template_from_X3DNA=True, temp_name='temp')`
Apply forcefield parameters to the DNA in the simulation system
This function finds out the bond, angle, stacking, and dihedrals potentials in the DNA chains. All the potentials it build are within the same ssDNA chain, so it does not analyze base pairing and cross-stacking interactions. 
This function will produce self.bonds, self.angles, self.stackings, and self.dihedrals (self means the object created from class DNA). 

`class BasePair`
The function form of BasePair is given by paper: J. Chem. Phys. 139, 144903 (2013). The potential is defined with `openmm.openmm.CustomHbondForce`, which is used to define hydrogen bonds. Class method `defineInteraction` defines the interactions between CG base atoms. 

`class CrossStacking`
The function form of cross-stacking is given by paper: J. Chem. Phys. 139, 144903 (2013). 

Note in ff3SPN2.py, `class Bond`, `class Angle`, `class Stacking`, `class Dihedral`, `class BasePair`, `class CrossStacking`, `class Exclusion` all inherit `class Force`. For `class Force`, in `__init__`, it runs `self.reset()` and `self.defineInteraction()`. `self.reset()` will define the potential function form, and `self.defineInteraction()` will add each individual interactions (i.e. each individual bond, or each individual angle, etc). 

The following dictionaries are defined in ff3SPN2.py for DNA-DNA interactions and DNA-protein interactions:

```
forces = dict(Bond=Bond,
              Angle=Angle,
              Stacking=Stacking,
              Dihedral=Dihedral,
              BasePair=BasePair,
              CrossStacking=CrossStacking,
              Exclusion=Exclusion,
              Electrostatics=Electrostatics)

protein_dna_forces=dict(ExclusionProteinDNA=ExclusionProteinDNA,
                        ElectrostaticsProteinDNA=ElectrostaticsProteinDNA)
```

Some notes about openSMOG3SPN2/openFiber.py

`buildNonBondedExclusionsList`
This function defines the exclusion list that includes all the pairs of atoms that should be excluded from: (a) DNA-DNA excluded volume potential; (b) DNA-DNA electrostatic interactions; (c) protein-protein nonbonded MJ potential and protein-protein electrostatic interactions; 
The exclusion list should be defined as: (1) for DNA, based on ff3SPN2.addNonBondedExclusions; (2) for protein, if the two protein CG atoms are within the same chain with 1-2 or 1-3 or 1-4 potentials, or if the two protein CG atoms are within smog output exclusion list.
Additionally, since in lammps, when we apply MJ potential with "lj/cut/coul/debye", it cannot add exclusions for protein native pairs (i.e. pairs included in smog output exclusion list), so when we compare openmm results with lammps outputs, in openmm we use the exclusion list that does not include native protein pairs. However, when we perform simulations with openmm, remember to include protein native pairs into the exclusion list. 
Note previously when we perform simulations for nucleosomes or chromatin fibers with lammps, we do not exclude protein native pairs from protein-protein nonbonded MJ potential or electrostatic interactions. Though in principle we should exclude protein native pairs, but since we use rigid body setting for histone core and core DNA, so whether exclude protein native pairs or not does not affect the result, so the previous simulations performed by lammps should be correct. Here, we aim to develop a general workflow that can be applied to any protein-DNA system, so to keep the settings general, we add protein native pairs into the exclusion list in openmm.  

# Test open3spn2 and smog
Basically we check if the protein-DNA system simulated by openmm has the same potential as simulated by lammps. 

Some important notes: 
(1) To run protein-DNA simulation on gpu, the exclusion list (i.e. the list of atom pairs that nonbonded interactions should be neglected) should be shared among all the nonbonded interactions
(2) Be careful that in ff3SPN2.py, there is a class called `Exclusion`, and such class represents a type of non-bonded interaction, so such interactions should also be removed for atom pairs within exclusion list
(3) The exclusion list should includes: (a) atom pairs that are within the same protein chain with 1-2, 1-3, 1-4 interactions; (b) atom pairs within smog output exclusion list, since SBM potentials exist between these atom pairs
(4) If two protein atoms are within the same rigid body, then we do not add them to the exclusion list, even if they satisfy the condition stated in (3)
(5) As ff3SPN2.py initialize DNA-DNA and protein-DNA excluded volume and electrostatic interaction exclusion list with exclusions between certain CG DNA atoms, so to keep things consistent, we do not remove CG DNA atom exclusions within the same rigid body from the overall exclusion list (i.e. we keep the exclusions between CG DNA atoms the same as what ff3SPN2.py initializes by function `addNonBondedExclusions` to avoid any possible bugs)



