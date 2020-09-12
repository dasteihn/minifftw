main_dir = $(shell pwd)

normal:
	python3 setup.py build --verbose

mpi:
	MFFTW_MPI=1 python3 setup.py build --verbose

lrz-cm2-normal:
	MFFTW_BASE=main_dir make -C ./clusters/lrz/CoolMUC-2 normal

lrz-cm2-mpi:
	MFFTW_BASE=main_dir make -C ./clusters/lrz/CoolMUC-2 mpi

clean:
	rm -rf build/
