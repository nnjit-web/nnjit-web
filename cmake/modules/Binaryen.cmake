
# Be compatible with older version of CMake
find_binaryen(${USE_BINARYEN})

if(USE_BINARYEN)
  if(NOT Binaryen_FOUND)
    message(FATAL_ERROR "Cannot find Binaryen, USE_BINARYEN=" ${USE_BINARYEN})
  endif()
  include_directories(SYSTEM ${Binaryen_INCLUDE_DIRS})
  message(STATUS "Build with Binaryen support")
  tvm_file_glob(GLOB COMPILER_BINARYEN_SRCS src/target/wasm/*.cc)
  list(APPEND COMPILER_SRCS ${COMPILER_BINARYEN_SRCS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${Binaryen_LIBRARY})
endif(USE_BINARYEN)
