cmake_minimum_required(VERSION 3.0)
project(red_detect)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O3 -std=c++14")
find_package(OpenCV REQUIRED)
#add_executable(red_detect detect_red.cpp)

if(HAVE_CUDA)
  set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_LIBRARIES}     ${CUDA_npp_LIBRARY})
  if(HAVE_CUBLAS)
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_cublas_LIBRARY})
  endif()
  if(HAVE_CUFFT)
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CUDA_cufft_LIBRARY})
  endif()
  foreach(p ${CUDA_LIBS_PATH})
	set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CMAKE_LIBRARY_PATH_FLAG}${p})
  endforeach()
  else()
endif()

add_executable(red_detect red_detect.cpp)
target_link_libraries(red_detect ${OpenCV_LIBS})
