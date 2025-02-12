# AX-B_SparseSolver_MPI

This code solves s system of linear equations AX=B using the MPI (Message Passing Interface) for distributed computing and PETSc (Portable, Extensible Toolkit for Scientific Computation) for efficient handling of sparse matrices.
This code initializes a sparse matrix A, a right-hand-side matrix B, and then uses KSP (Krylov Subspace Methods) from PETSc to solve for 
X in parallel.

Requirements:
Install MPI (e.g., OpenMPI or MPICH)
Install PETSc (with MPI support)
Compile with mpif90 and link with PETSc libraries
