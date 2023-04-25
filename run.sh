# build the system
#n_nucl=2
#python build_mono_fiber.py --n_nucl ${n_nucl} --mode default --run_smog

n_nucl=12
dna_seq_file=/Users/administrator/Documents/Projects/CA_SBM_3SPN2C_OPENMM/data/chromatin-12mer/dnaSeq.txt
python build_mono_fiber.py --n_nucl ${n_nucl} --mode default --dna_seq_file ${dna_seq_file} --run_smog

