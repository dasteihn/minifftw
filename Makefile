normal:
	meson build && ninja -C build
	cp ./build/*.so .

mpi:
	meson build -Dmpi=true && ninja -C build
	cp ./build/*.so .

lrz-cm2-normal:
	MFFTW_BASE=$(PWD) $(MAKE) -C ./clusters/lrz/CoolMUC-2 normal
	cp ./clusters/lrz/CoolMUC-2/build/lib*/*.so .

lrz-cm2-mpi:
	MFFTW_BASE=$(PWD) $(MAKE) -C ./clusters/lrz/CoolMUC-2 mpi
	cp ./clusters/lrz/CoolMUC-2/build/lib*/*.so .

clean:
	rm -f *.so
	rm -rf build/
	rm -rf clusters/lrz/CoolMUC-2/build/
