normal:
	python3 setup.py build --verbose

mpi:
	MFFTW_MPI=1 python3 setup.py build --verbose
	rm ./build/lib*/*
	mpiicc -pthread -shared -B \
		$(CONDA_PREFIX)/compiler_compat \
		-L$(CONDA_PREFIX)/lib \
		-Wl,-rpath=~/.conda/envs/$(CONDA_PREFIX)/lib \
		-Wl,--no-as-needed \
		-Wl,--sysroot=/ $(shell find build/ -name minifftw.o) \
		$(shell find build util.o) $(shell find build -name plancapsule.o) \
		-lfftw3 \
		-o build/lib.linux-x86_64-3.8/minifftw.cpython-38-x86_64-linux-gnu.so \
		$(FFTW_INC) -lpthread -lfftw3_threads -lfftw3_mpi
