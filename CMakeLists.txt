cmake_minimum_required(VERSION 3.0)
project(base)
set (CMAKE_CXX_STANDARD 11)

include_directories(
  lib/
  common/
)

link_directories(
	${PROJECT_SOURCE_DIR}/OpenCL-ICD-Loader/inc
	${PROJECT_SOURCE_DIR}/OpenCL-ICD-Loader/inc/CL
)

IF (WIN32)
  SET(CL_LIBRARIES OpenCL)
ELSE (WIN32)
  SET(CL_LIBRARIES OpenCL)
ENDIF(WIN32)

add_definitions(-DSRC_PATH=\"${CMAKE_SOURCE_DIR}/\")

# Function to create OpenCL executables
function(create_opencl_executable name src)
    add_executable(${name} ${src} common/clutils.cpp)
    if(APPLE)
        target_link_libraries(${name} "-framework OpenCL")
    else()
        target_link_libraries(${name} ${CL_LIBRARIES})
    endif()
endfunction()

# Set varaibles
#set(project_1 "opencl-base-code")
#set(project_2 "Conway_Game_of_Life")
set(project_3 "Parallel_scan")
set(project_4 "Parallel_closest_pair_of_points")

# Create executables using the function
#create_opencl_executable(base1 "${project_1}/base.cpp;${project_1}/base.cl")
#create_opencl_executable(base2 "${project_2}/base.cpp;${project_2}/base.cl")
create_opencl_executable(ex1 "${project_3}/base.cpp;${project_3}/base.cl")
create_opencl_executable(ex2 "${project_4}/base.cpp;${project_4}/base.cl")

add_subdirectory(OpenCL-ICD-Loader)
