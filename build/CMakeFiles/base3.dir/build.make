# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/asus/Documents/OpenCL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/asus/Documents/OpenCL/build

# Include any dependencies generated for this target.
include CMakeFiles/base3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/base3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/base3.dir/flags.make

CMakeFiles/base3.dir/Parallel_scan/base.cpp.o: CMakeFiles/base3.dir/flags.make
CMakeFiles/base3.dir/Parallel_scan/base.cpp.o: ../Parallel_scan/base.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/asus/Documents/OpenCL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/base3.dir/Parallel_scan/base.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/base3.dir/Parallel_scan/base.cpp.o -c /home/asus/Documents/OpenCL/Parallel_scan/base.cpp

CMakeFiles/base3.dir/Parallel_scan/base.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/base3.dir/Parallel_scan/base.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asus/Documents/OpenCL/Parallel_scan/base.cpp > CMakeFiles/base3.dir/Parallel_scan/base.cpp.i

CMakeFiles/base3.dir/Parallel_scan/base.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/base3.dir/Parallel_scan/base.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asus/Documents/OpenCL/Parallel_scan/base.cpp -o CMakeFiles/base3.dir/Parallel_scan/base.cpp.s

CMakeFiles/base3.dir/common/clutils.cpp.o: CMakeFiles/base3.dir/flags.make
CMakeFiles/base3.dir/common/clutils.cpp.o: ../common/clutils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/asus/Documents/OpenCL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/base3.dir/common/clutils.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/base3.dir/common/clutils.cpp.o -c /home/asus/Documents/OpenCL/common/clutils.cpp

CMakeFiles/base3.dir/common/clutils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/base3.dir/common/clutils.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asus/Documents/OpenCL/common/clutils.cpp > CMakeFiles/base3.dir/common/clutils.cpp.i

CMakeFiles/base3.dir/common/clutils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/base3.dir/common/clutils.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asus/Documents/OpenCL/common/clutils.cpp -o CMakeFiles/base3.dir/common/clutils.cpp.s

# Object files for target base3
base3_OBJECTS = \
"CMakeFiles/base3.dir/Parallel_scan/base.cpp.o" \
"CMakeFiles/base3.dir/common/clutils.cpp.o"

# External object files for target base3
base3_EXTERNAL_OBJECTS =

base3: CMakeFiles/base3.dir/Parallel_scan/base.cpp.o
base3: CMakeFiles/base3.dir/common/clutils.cpp.o
base3: CMakeFiles/base3.dir/build.make
base3: OpenCL-ICD-Loader/libOpenCL.so.1.2
base3: CMakeFiles/base3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/asus/Documents/OpenCL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable base3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/base3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/base3.dir/build: base3

.PHONY : CMakeFiles/base3.dir/build

CMakeFiles/base3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/base3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/base3.dir/clean

CMakeFiles/base3.dir/depend:
	cd /home/asus/Documents/OpenCL/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/asus/Documents/OpenCL /home/asus/Documents/OpenCL /home/asus/Documents/OpenCL/build /home/asus/Documents/OpenCL/build /home/asus/Documents/OpenCL/build/CMakeFiles/base3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/base3.dir/depend

