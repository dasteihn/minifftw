project(
  'minifftw', 'c',
  version : '1',
  license : 'GPL',
  meson_version : '>= 0.47',
)

pymod = import('python')
py = pymod.find_installation('python3', required : true)
c_srcs = ['minifftw.c', 'plancapsule.c', 'util.c']

# numpy dependency consist of just includes
message('searching numpy include dirs')
numpy_cmd = run_command(
  'python3',
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true,
)
numpy_dep = declare_dependency(
  include_directories : include_directories(numpy_cmd.stdout().strip())
)

cc = meson.get_compiler('c')
minifftw_c_args = []
minifftw_link_args = ['-lfftw3_threads']
minifftw_deps = [
  dependency('threads'),
  cc.find_library('m', required : true),
  py.dependency(),
  numpy_dep,
]

if get_option('mpi')
  minifftw_c_args += '-DMFFTW_MPI'
  minifftw_link_args += '-lfftw3_mpi'
  minifftw_deps += dependency('mpi', language: 'c')
endif

py.extension_module(
  'minifftw',
  c_srcs,
  dependencies : minifftw_deps,
  c_args : minifftw_c_args,
  link_args : minifftw_link_args,
)

