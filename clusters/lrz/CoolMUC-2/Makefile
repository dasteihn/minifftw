normal:
	python3 ./clusters/lrz/setup.py build --verbose

mpi:
	MFFTW_MPI=1 python3 ./clusters/lrz/setup.py build --verbose
	rm ./build/lib*/*
	mpiicc -pthread -shared -B \
		/dss/dsshome1/lxc0D/ka76ges/.conda/envs/py38/compiler_compat \
		-L/dss/dsshome1/lxc0D/ka76ges/.conda/envs/py38/lib \
		-Wl,-rpath=/dss/dsshome1/lxc0D/ka76ges/.conda/envs/py38/lib \
		-Wl,--no-as-needed \
		-Wl,--sysroot=/ build/temp.linux-x86_64-3.8/minifftw.o build/temp.linux-x86_64-3.8/util.o \
		build/temp.linux-x86_64-3.8/plancapsule.o -lfftw3 \
		-o build/lib.linux-x86_64-3.8/minifftw.cpython-38-x86_64-linux-gnu.so \
		-I/dss/dsshome1/lrz/sys/spack/staging/20.1.1/opt/haswell/fftw/3.3.8-intel-y3t2y3j/include \
		-lpthread -lfftw3_threads -lfftw3_mpi
