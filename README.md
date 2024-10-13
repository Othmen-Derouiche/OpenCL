# OpenCL

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