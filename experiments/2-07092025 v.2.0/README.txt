ZFSC Project (modular version with dark level)
------------------------------------------------
Files:
- zfsc_predictor.py : main CLI file, uses compose_matrix
- geometry.py       : base 'onion' + gravity node + dark matter node, compose_matrix
- fibers_spin.py    : SU(2) spin fiber
- fibers_color.py   : SU(3)+anticolor fiber
- fibers_isospin.py : weak isospin fiber
- fibers_charge.py  : charge shift fiber
- doubling_nambu.py : Nambu doubling for antiparticles

Global flags and parameters are defined at the top of each file.
Example usage in zfsc_predictor.py:
    M = compose_matrix(args.matrix_size, d, r)

New feature:
- ENABLE_DARK (in geometry.py) adds a 'dark matter' top-level node
- Energy fractions (visible/dark/grav) are printed at the end of run
