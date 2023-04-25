mol new 1kx5.pdb
mol new fiber-167-12_chain_id_changed.pdb

set length 167
set all0 [atomselect 0 "all"]
set sel0 [atomselect 0 "chain I and name P and resid >= -72 and resid <= 72"]
set start [expr 167 - 144]
set end [expr 167]

set sel1 [atomselect 1 "chain A and name P and resid >= ${start} and resid <= ${end}"]
set m1 [measure fit $sel0 $sel1]
$all0 move $m1
set histone1 [atomselect 0 "not chain I and not chain J"]
$histone1 writepdb histone-1.pdb
measure rmsd $sel0 $sel1

set sel2 [atomselect 1 "chain C and name P and resid >= ${start} and resid <= ${end}"]
set m2 [measure fit $sel0 $sel2]
$all0 move $m2
set histone2 [atomselect 0 "not chain I and not chain J"]
$histone2 writepdb histone-2.pdb
measure rmsd $sel0 $sel2

set sel3 [atomselect 1 "chain E and name P and resid >= ${start} and resid <= ${end}"]
set m3 [measure fit $sel0 $sel3]
$all0 move $m3
set histone3 [atomselect 0 "not chain I and not chain J"]
$histone3 writepdb histone-3.pdb
measure rmsd $sel0 $sel3

set sel4 [atomselect 1 "chain G and name P and resid >= ${start} and resid <= ${end}"]
set m4 [measure fit $sel0 $sel4]
$all0 move $m4
set histone4 [atomselect 0 "not chain I and not chain J"]
$histone4 writepdb histone-4.pdb
measure rmsd $sel0 $sel4

set sel5 [atomselect 1 "chain I and name P and resid >= ${start} and resid <= ${end}"]
set m5 [measure fit $sel0 $sel5]
$all0 move $m5
set histone5 [atomselect 0 "not chain I and not chain J"]
$histone5 writepdb histone-5.pdb
measure rmsd $sel0 $sel5

set sel6 [atomselect 1 "chain K and name P and resid >= ${start} and resid <= ${end}"]
set m6 [measure fit $sel0 $sel6]
$all0 move $m6
set histone6 [atomselect 0 "not chain I and not chain J"]
$histone6 writepdb histone-6.pdb
measure rmsd $sel0 $sel6

set sel7 [atomselect 1 "chain M and name P and resid >= ${start} and resid <= ${end}"]
set m7 [measure fit $sel0 $sel7]
$all0 move $m7
set histone7 [atomselect 0 "not chain I and not chain J"]
$histone7 writepdb histone-7.pdb
measure rmsd $sel0 $sel7

set sel8 [atomselect 1 "chain O and name P and resid >= ${start} and resid <= ${end}"]
set m8 [measure fit $sel0 $sel8]
$all0 move $m8
set histone8 [atomselect 0 "not chain I and not chain J"]
$histone8 writepdb histone-8.pdb
measure rmsd $sel0 $sel8

set sel9 [atomselect 1 "chain Q and name P and resid >= ${start} and resid <= ${end}"]
set m9 [measure fit $sel0 $sel9]
$all0 move $m9
set histone9 [atomselect 0 "not chain I and not chain J"]
$histone9 writepdb histone-9.pdb
measure rmsd $sel0 $sel9

set sel10 [atomselect 1 "chain S and name P and resid >= ${start} and resid <= ${end}"]
set m10 [measure fit $sel0 $sel10]
$all0 move $m10
set histone10 [atomselect 0 "not chain I and not chain J"]
$histone10 writepdb histone-10.pdb
measure rmsd $sel0 $sel10

set sel11 [atomselect 1 "chain U and name P and resid >= ${start} and resid <= ${end}"]
set m11 [measure fit $sel0 $sel11]
$all0 move $m11
set histone11 [atomselect 0 "not chain I and not chain J"]
$histone11 writepdb histone-11.pdb
measure rmsd $sel0 $sel11

set sel12 [atomselect 1 "chain W and name P and resid >= ${start} and resid <= ${end}"]
set m12 [measure fit $sel0 $sel12]
$all0 move $m12
set histone12 [atomselect 0 "not chain I and not chain J"]
$histone12 writepdb histone-12.pdb
measure rmsd $sel0 $sel12

exit