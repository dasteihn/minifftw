normal:
	python3 setup.py build --verbose
	cp ./build/lib*/*.so .

mpi:
	MFFTW_MPI=1 python3 setup.py build --verbose
	cp ./build/lib*/*.so .

lrz-cm2-normal:
	MFFTW_BASE=$(PWD) $(MAKE) -C ./clusters/lrz/CoolMUC-2 normal
	cp ./clusters/lrz/CoolMUC-2/build/lib*/*.so .

lrz-cm2-mpi:
	MFFTW_BASE=$(PWD) $(MAKE) -C ./clusters/lrz/CoolMUC-2 mpi
	cp ./clusters/lrz/CoolMUC-2/build/lib*/*.so .

clean:
	rm *.so || true
	rm -rf build/
	rm -rf clusters/lrz/CoolMUC-2/build/
