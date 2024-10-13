# OpenCL Programming Guide

## Introduction to OpenCL

OpenCL (Open Computing Language) is a framework for developing programs that can be executed across heterogeneous platforms, including CPUs, GPUs, and other processors. Below, we explain the key concepts and components of OpenCL.

### Key Concepts of OpenCL Architecture

1. **Host and Device**
   - **Host**: The CPU and associated memory where the program runs and manages the execution of OpenCL kernels.
   - **Device**: The hardware that executes OpenCL kernels (e.g., CPUs, GPUs, FPGAs, and accelerators).

2. **Platform**
   - An OpenCL platform consists of a host and one or more devices. It facilitates the execution of kernels across different devices.

3. **Context**
   - A context is a runtime environment where OpenCL kernels execute. It includes all devices, memory, and resources needed for kernel execution.

4. **Command Queue**
   - The command queue is used by the host to issue commands (such as kernel execution and memory operations) to a device. It holds commands in the order they are issued, enabling asynchronous execution.

5. **Kernel**
   - A kernel is a function that runs on the device and serves as the entry point for parallel execution. Each kernel can be concurrently executed across multiple data elements.

6. **Work Item and Work Group**
   - **Work Item**: An instance of a kernel. Each work item is assigned a unique ID to differentiate it from other instances.
   - **Work Group**: A group of work items that execute a kernel together on a single compute unit, allowing communication through local memory.

7. **Memory Model**
   - OpenCL utilizes different types of memory:
     - **Global Memory**: Accessible by all work items across devices; large but high latency.
     - **Local Memory**: Shared among work items within a work group; lower latency than global memory.
     - **Private Memory**: Local to a single work item.

8. **Barriers**
   - Barriers are synchronization points within a work group, ensuring that all work items reach a particular execution point before proceeding.

---
###  regenerate the Makefile using CMake
mkdir -p build
cd build
# Execute CMake from the build directory
cmake ..
# Compile the Makefile
make
# Clean the previous build 
make clean
# or delete all the contents of the build directory to ensure a fresh start
rm -rf *

### Example : perform the sum of two vector C=A+B in OpenCL
- Check for OpenCL-capable device(s);
- Memory allocation on the device;
- Data transfer to the device;
- Retrieve data from the device;
- Compile C/C++ programs that launch OpenCL kernels.
## Initialisation 
# Initially, OpenCL needs to initialize a context
Contexts are used by the OpenCL runtime for managing command queues,
memory, and for executing kernels on one or more devices connected to the
the context
cl::Context context(CL_DEVICE_TYPE_DEFAULT);
# Then, it is required to initialize a queue : Push commands to the device
cl::CommandQueue queue(context);
## Allocating Memory on the Device
# create three buffers
cl::Buffer A_d(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
cl::Buffer B_d(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
cl::Buffer C_d(context, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE);
# write the buffers to the queue to send their values from the host to the device
queue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
queue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(int) * SIZE, B_h);
## the kernel is provided as a string:
- Kernels need to return the void type.
- The global keyword means that the buffer lies on the global memory
std::string kernel_code =
"
void kernel simple_add(global const int* A, global const int* B, global int* C){ "
"
C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];
"
"
}
## Building the Kernel
# append the kernel string to the program source
cl::Program::Sources sources;
sources.push_back({ kernel_code.c_str(),kernel_code.length() });
# create a program object using the program source as argument
- the program contains the simple_add kernel
cl::Program program(context, sources); 
# instantiate a kernel object for execution with three cl:buffers as arguments
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> simple_add(cl::Kernel(program, "simple_add"));
# invoke the kernel simple_add
cl::NDRange global(SIZE);
simple_add(cl::EnqueueArgs(queue, global), A_d, B_d, C_d).wait();
# retrieve the result by using the queue
queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * SIZE, C);