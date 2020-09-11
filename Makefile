normal:
	python3 setup.py build --verbose

mpi:
	MFFTW_MPI=1 python3 setup.py build --verbose

lrz:
	python3 ./clusters/lrz/setup.py build --verbose

lrz-mpi:
	MFFTW_MPI=1 python3 ./clusters/lrz/setup.py build --verbose

clean:
	rm -rf build/
