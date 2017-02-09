cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if (${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR
            "Do not build in-source. Delete the CMakeFiles and CMakeCache.txt \
             which have been created in the src folder. Afterwards, create a \
             subfolder within ../cmake_workspace and call cmake ../../src from \
             there.")
endif()

project(cub_hip LANGUAGES CXX)

#------------------------------------HIP---------------------------------------#
find_program(hipcc NAMES hipcc
                   PATHS /home/alexv/Programming/HIP_private_github/build/install/bin)
                         #/opt/rocm/bin)
set(CMAKE_CXX_COMPILER ${hipcc})

add_library(hip INTERFACE)
target_include_directories(hip SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/boost
                                                ${CMAKE_CURRENT_SOURCE_DIR}/external/bolt/include
                                                /home/alexv/Programming/HIP_private_github/build/install/include)
                                                #/opt/rocm/include/hip)
link_directories(/home/alexv/Programming/HCC/cmake-build-release/lib
                 /home/alexv/Programming/HIP_private_github/build/install/lib)
target_compile_options(hip BEFORE INTERFACE -v)
#target_compile_options(hcc BEFORE INTERFACE -std=c++amp -stdlib=libc++)
#target_link_libraries(hcc INTERFACE -std=c++amp c++ c++abi m mcwamp dl)
add_library(external::hip ALIAS hip)
#-------------------------------------HIP--------------------------------------#

#------------------------------------Tests-------------------------------------#
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test/link_a.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/link_b.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_allocator.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_histogram.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_load_store.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_radix_sort.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_reduce.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_scan.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_histogram.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_radix_sort.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_reduce.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_reduce_by_key.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_run_length_encode.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_scan.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_select_if.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_select_unique.cpp
#                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_grid_barrier.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_iterator.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_warp_reduce.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_warp_scan.cpp
                            PROPERTIES LANGUAGE CXX)

add_executable(link ${CMAKE_CURRENT_SOURCE_DIR}/test/link_a.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/link_b.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/link_main.cpp)
target_include_directories(link PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(link PUBLIC external::hip)

add_executable(test_allocator ${CMAKE_CURRENT_SOURCE_DIR}/test/test_allocator.cpp)
target_include_directories(test_allocator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_allocator PUBLIC external::hip)

add_executable(test_block_histogram ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_histogram.cpp)
target_include_directories(test_block_histogram PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_histogram PUBLIC external::hip)

add_executable(test_block_load_store ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_load_store.cpp)
target_include_directories(test_block_load_store PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_load_store PUBLIC external::hip)

add_executable(test_block_radix_sort ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_radix_sort.cpp)
target_include_directories(test_block_radix_sort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_radix_sort PUBLIC external::hip)

add_executable(test_block_reduce ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_reduce.cpp)
target_include_directories(test_block_reduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_reduce PUBLIC external::hip)
target_compile_definitions(test_block_reduce PUBLIC TEST_ARCH=620
                                                    TEST_RAKING)
                                                    # TODO: this triggers a compiler bug.
                                                    #TEST_WARP_REDUCTIONS)

add_executable(test_block_scan ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_scan.cpp)
target_include_directories(test_block_scan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_scan PUBLIC external::hip)
target_compile_definitions(test_block_scan PUBLIC TEST_RAKING
                                                  TEST_RAKING_MEMOIZE
                                                  TEST_WARP_SCANS)

# TODO: the histogram test uses NPP, for which we do not have an equivalent.
#add_executable(test_device_histogram ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_histogram.cu)
#target_include_directories(test_device_histogram PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(test_device_histogram PUBLIC external::hip)

add_executable(test_device_radix_sort ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_radix_sort.cpp)
target_include_directories(test_device_radix_sort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_radix_sort PUBLIC external::hip)

add_executable(test_device_reduce ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_reduce.cpp)
target_include_directories(test_device_reduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_reduce PUBLIC external::hip)

# TODO: the device reduce test needs vector types which are TotallyOrdered, and
#       HIP's do not match this requirement.
add_executable(test_device_reduce_by_key ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_reduce_by_key.cpp)
target_include_directories(test_device_reduce_by_key PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_reduce_by_key PUBLIC external::hip)

add_executable(test_device_run_length_encode ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_run_length_encode.cpp)
target_include_directories(test_device_run_length_encode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_run_length_encode PUBLIC external::hip)

# TODO: the device scan test needs vector types which are TotallyOrdered, and
#       HIP's do not match this requirement.
add_executable(test_device_scan ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_scan.cpp)
target_include_directories(test_device_scan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_scan PUBLIC external::hip)

# TODO: the select_if test needs vector types which are TotallyOrdered, and
#       HIP's do not match this requirement.
add_executable(test_device_select_if ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_select_if.cpp)
target_include_directories(test_device_select_if PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_select_if PUBLIC external::hip)

add_executable(test_device_select_unique ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_select_unique.cpp)
target_include_directories(test_device_select_unique PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_select_unique PUBLIC external::hip)

# TODO: the grid barrier test requires __threadfence and __threadfence_block.
#add_executable(test_grid_barrier ${CMAKE_CURRENT_SOURCE_DIR}/test/test_grid_barrier.cpp)
#target_include_directories(test_grid_barrier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(test_grid_barrier PUBLIC external::hip)

add_executable(test_iterator ${CMAKE_CURRENT_SOURCE_DIR}/test/test_iterator.cpp)
target_include_directories(test_iterator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_iterator PUBLIC external::hip)

add_executable(test_warp_reduce ${CMAKE_CURRENT_SOURCE_DIR}/test/test_warp_reduce.cpp)
target_include_directories(test_warp_reduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_warp_reduce PUBLIC external::hip)

# TODO: this is temporarily disabled, as it causes HCC breakage.
add_executable(test_warp_scan ${CMAKE_CURRENT_SOURCE_DIR}/test/test_warp_scan.cpp)
target_include_directories(test_warp_scan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_warp_scan PUBLIC external::hip)
