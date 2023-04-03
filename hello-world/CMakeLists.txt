configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/x.mtx data/x.mtx COPYONLY)
configure_file(data/y.mtx data/y.mtx COPYONLY)

add_executable(hello-world hello-world.cpp)
target_link_libraries(hello-world Ginkgo::ginkgo)

add_executable(read-matrix read-matrix.cpp)
target_link_libraries(read-matrix Ginkgo::ginkgo)

add_executable(matrix-assembly matrix-assembly.cpp)
target_link_libraries(matrix-assembly Ginkgo::ginkgo)

add_executable(solve solve.cpp)
target_link_libraries(solve Ginkgo::ginkgo)
