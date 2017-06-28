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
find_program(hipcc NAMES hipcc PATHS /opt/rocm/bin)
set(CMAKE_CXX_COMPILER ${hipcc})

add_library(hip INTERFACE)
target_include_directories(
    hip SYSTEM INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/boost
        ${CMAKE_CURRENT_SOURCE_DIR}/external/bolt/include
        /opt/rocm/include/hip)
target_link_libraries(hip INTERFACE rt)
link_directories(/opt/rocm/lib)
add_library(external::hip ALIAS hip)
#-------------------------------------HIP--------------------------------------#

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/block/example_block_radix_sort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/block/example_block_reduce.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/block/example_block_scan.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/block/reduce_by_key.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_partition_flagged.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_partition_if.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_radix_sort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_reduce.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_scan.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_select_flagged.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_select_if.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_select_unique.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_sort_find_non_trivial_runs.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental/defunct/example_coo_spmv.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental/defunct/test_device_seq_reduce.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental/histogram_compare.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental/spmv_compare.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_grid_barrier.cu
    PROPERTIES LANGUAGE CXX)

enable_testing()
#----------------------------------Examples------------------------------------#
#------------------------------------Block-------------------------------------#
add_executable(
    example_block_radix_sort
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/block/example_block_radix_sort.cu)
target_include_directories(
    example_block_radix_sort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_block_radix_sort PUBLIC external::hip)
add_test(
    NAME example_block_radix_sort
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_block_radix_sort)
set_property(
    TEST example_block_radix_sort
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    example_block_reduce
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/block/example_block_reduce.cu)
target_include_directories(
    example_block_reduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_block_reduce PUBLIC external::hip)
add_test(
    NAME example_block_reduce
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_block_reduce)
set_property(
    TEST example_block_reduce
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    example_block_scan
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/block/example_block_scan.cu)
target_include_directories(
    example_block_scan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_block_scan PUBLIC external::hip)
add_test(
    NAME example_block_scan
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_block_scan)
set_property(
    TEST example_block_scan
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")
#------------------------------------Block-------------------------------------#
#-----------------------------------Device-------------------------------------#
add_executable(
    example_device_partition_flagged
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_partition_flagged.cu)
target_include_directories(
    example_device_partition_flagged PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_device_partition_flagged PUBLIC external::hip)
add_test(
    NAME example_device_partition_flagged
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_device_partition_flagged)
set_property(
    TEST example_device_partition_flagged
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    example_device_partition_if
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_partition_if.cu)
target_include_directories(
    example_device_partition_if PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_device_partition_if PUBLIC external::hip)
add_test(
    NAME example_device_partition_if
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_device_partition_if)
set_property(
    TEST example_device_partition_if
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    example_device_radix_sort
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_radix_sort.cu)
target_include_directories(
    example_device_radix_sort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_device_radix_sort PUBLIC external::hip)
add_test(
    NAME example_device_radix_sort
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_device_radix_sort)
set_property(
    TEST example_device_radix_sort
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    example_device_reduce
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_reduce.cu)
target_include_directories(
    example_device_reduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_device_reduce PUBLIC external::hip)
add_test(
    NAME example_device_reduce
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_device_reduce)
set_property(
    TEST example_device_reduce
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    example_device_select_flagged
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_select_flagged.cu)
target_include_directories(
    example_device_select_flagged PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_device_select_flagged PUBLIC external::hip)
add_test(
    NAME example_device_select_flagged
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_device_select_flagged)
set_property(
    TEST example_device_select_flagged
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    example_device_select_if
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_select_if.cu)
target_include_directories(
    example_device_select_if PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_device_select_if PUBLIC external::hip)
add_test(
    NAME example_device_select_if
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_device_select_if)
set_property(
    TEST example_device_select_if
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    example_device_select_unique
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_select_unique.cu)
target_include_directories(
    example_device_select_unique PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example_device_select_unique PUBLIC external::hip)
add_test(
    NAME example_device_select_unique
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_device_select_unique)
set_property(
    TEST example_device_select_unique
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    example_device_sort_find_non_trivial_runs
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_sort_find_non_trivial_runs.cu)
target_include_directories(
    example_device_sort_find_non_trivial_runs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    example_device_sort_find_non_trivial_runs PUBLIC external::hip)
add_test(
    NAME example_device_sort_find_non_trivial_runs
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/example_device_sort_find_non_trivial_runs)
set_property(
    TEST example_device_sort_find_non_trivial_runs
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")
#-----------------------------------Device-------------------------------------#
#----------------------------------Examples------------------------------------#

#--------------------------------Experimental----------------------------------#
add_executable(
    histogram_compare
        ${CMAKE_CURRENT_SOURCE_DIR}/experimental/histogram_compare.cu)
target_include_directories(
    histogram_compare PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(histogram_compare PUBLIC external::hip)
add_test(
    NAME histogram_compare
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/histogram_compare)
set_property(
    TEST histogram_compare
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

# TODO: this uses cuSparse, replacement with hcSparse should / could be viable.
#add_executable(
#    spmv_compare
#        ${CMAKE_CURRENT_SOURCE_DIR}/experimental/spmv_compare.cu)
#target_include_directories(
#    spmv_compare PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(spmv_compare PUBLIC external::hip)
#add_test(
#    NAME spmv_compare
#    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/spmv_compare)
#set_property(
#    TEST spmv_compare
#    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")
#--------------------------------Experimental----------------------------------#

#------------------------------------Tests-------------------------------------#
add_executable(
    link
        ${CMAKE_CURRENT_SOURCE_DIR}/test/link_a.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/link_b.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/link_main.cpp)
target_include_directories(link PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(link PUBLIC external::hip)
add_test(
    NAME link
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/link)
set_property(
    TEST link
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    test_allocator ${CMAKE_CURRENT_SOURCE_DIR}/test/test_allocator.cpp)
target_include_directories(test_allocator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_allocator PUBLIC external::hip)
add_test(
    NAME test_allocator
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_allocator)
set_property(
    TEST test_allocator
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    test_block_histogram
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_histogram.cpp)
target_include_directories(
    test_block_histogram PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_histogram PUBLIC external::hip)
add_test(
    NAME test_block_histogram
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_block_histogram)
set_property(
    TEST test_block_histogram
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    test_block_load_store
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_load_store.cpp)
target_include_directories(
    test_block_load_store PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_load_store PUBLIC external::hip)
add_test(
    NAME test_block_load_store
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_block_load_store)
set_property(
    TEST test_block_load_store
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    test_block_radix_sort
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_radix_sort.cpp)
target_include_directories(
    test_block_radix_sort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_radix_sort PUBLIC external::hip)
add_test(
    NAME test_block_radix_sort
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_block_radix_sort)
set_property(
    TEST test_block_radix_sort
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

# TODO: 27/06/2017 TestFoo test fails in SROA pass.
#       char{2, 4} and short{2, 4} cause invalid addrspacecast errors.
add_executable(
    test_block_reduce ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_reduce.cpp)
target_include_directories(test_block_reduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_reduce PUBLIC external::hip)
target_compile_definitions(
    test_block_reduce PUBLIC TEST_ARCH=620 TEST_RAKING TEST_WARP_REDUCTIONS)
add_test(
    NAME test_block_reduce
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_block_reduce)
set_property(
    TEST test_block_reduce
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    test_block_scan ${CMAKE_CURRENT_SOURCE_DIR}/test/test_block_scan.cpp)
target_include_directories(test_block_scan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_block_scan PUBLIC external::hip)
target_compile_definitions(
    test_block_scan PUBLIC TEST_RAKING TEST_RAKING_MEMOIZE TEST_WARP_SCANS)
add_test(
    NAME test_block_scan
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_block_scan)
set_property(
    TEST test_block_scan
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

# TODO: the histogram test uses NPP, for which we do not have an equivalent.
#add_executable(test_device_histogram ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_histogram.cpp)
#target_include_directories(test_device_histogram PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(test_device_histogram PUBLIC external::hip)
#add_test(
#    NAME test_device_histogram
#    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_device_histogram)
#set_property(
#    TEST test_device_histogram
#    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    test_device_radix_sort
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_radix_sort.cpp)
target_include_directories(
    test_device_radix_sort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_radix_sort PUBLIC external::hip)
add_test(
    NAME test_device_radix_sort
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_device_radix_sort)
set_property(
    TEST test_device_radix_sort
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

# TODO: 27/06/2017 - fails in InferAddressSpaces.
#add_executable(
#    test_device_reduce ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_reduce.cpp)
#target_include_directories(
#    test_device_reduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(test_device_reduce PUBLIC external::hip)
#add_test(
#    NAME test_device_reduce
#    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_device_reduce)
#set_property(
#    TEST test_device_reduce
#    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

# TODO: 27/06/2017 - invalid addrspacecast error.
#add_executable(
#    test_device_reduce_by_key
#        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_reduce_by_key.cpp)
#target_include_directories(
#    test_device_reduce_by_key PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(test_device_reduce_by_key PUBLIC external::hip)
#add_test(
#    NAME test_device_reduce_by_key
#    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_device_reduce_by_key)
#set_property(
#    TEST test_device_reduce_by_key
#    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

# TODO: 27/06/2017 - invalid addrspacecast error.
#add_executable(
#    test_device_run_length_encode
#        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_run_length_encode.cpp)
#target_include_directories(
#    test_device_run_length_encode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(test_device_run_length_encode PUBLIC external::hip)
#add_test(
#    NAME test_device_run_length_encode
#    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_device_run_length_encode)
#set_property(
#    TEST test_device_run_length_encode
#    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

# TODO: 27/06/2017 TestFoo test fails in SROA pass.
#       (u)char{2, 4} and ushort2 cause invalid addrspacecast errors.
add_executable(
    test_device_scan ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_scan.cpp)
target_include_directories(test_device_scan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_scan PUBLIC external::hip)
add_test(
    NAME test_device_scan
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_device_scan)
set_property(
    TEST test_device_scan
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

# TODO: 27/06/2017 TestFoo test fails in SROA pass.
add_executable(
    test_device_select_if
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_select_if.cpp)
target_include_directories(
    test_device_select_if PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_select_if PUBLIC external::hip)
add_test(
    NAME test_device_select_if
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_device_select_if)
set_property(
    TEST test_device_select_if
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

# TODO: 27/06/2017 TestFoo test fails in SROA pass.
add_executable(
    test_device_select_unique
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_device_select_unique.cpp)
target_include_directories(
    test_device_select_unique PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_device_select_unique PUBLIC external::hip)
add_test(
    NAME test_device_select_unique
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_device_select_unique)
set_property(
    TEST test_device_select_unique
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    test_grid_barrier ${CMAKE_CURRENT_SOURCE_DIR}/test/test_grid_barrier.cu)
target_include_directories(test_grid_barrier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_grid_barrier PUBLIC external::hip)
add_test(
    NAME test_grid_barrier
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_grid_barrier)
set_property(
    TEST test_grid_barrier
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(test_iterator ${CMAKE_CURRENT_SOURCE_DIR}/test/test_iterator.cpp)
target_include_directories(test_iterator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_iterator PUBLIC external::hip)
add_test(
    NAME test_iterator
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_iterator)
set_property(
    TEST test_iterator
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    test_warp_reduce ${CMAKE_CURRENT_SOURCE_DIR}/test/test_warp_reduce.cpp)
target_include_directories(test_warp_reduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_warp_reduce PUBLIC external::hip)
add_test(
    NAME test_warp_reduce
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_warp_reduce)
set_property(
    TEST test_warp_reduce
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")

add_executable(
    test_warp_scan ${CMAKE_CURRENT_SOURCE_DIR}/test/test_warp_scan.cpp)
target_include_directories(test_warp_scan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_warp_scan PUBLIC external::hip)
add_test(
    NAME test_warp_scan
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_warp_scan)
set_property(
    TEST test_warp_scan
    PROPERTY FAIL_REGULAR_EXPRESSION "fail;FAIL;error")
