all:
	python3 setup.py build --verbose

mpi:
	MFFTW_MPI=1 python3 setup.py build --verbose
