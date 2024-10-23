    // ----------------------------------------------------------

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <chrono>
#include <thread>
#include <vector>

using namespace std;
#define ARRAY_SIZE 9  // Size of the input array
// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------
// Sequential CPU version of the scan algorithm (prefix sum)
void cpu_scan(const vector<int>& input, vector<int>& output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

// Function to print an array
void printArray(const vector<int>& arr, const string& label) {
    cout << label << ": ";
    for (const auto& val : arr) {
        cout << val << " ";
    }
    cout << endl;
}

int main(int argc, char **argv) {
    const int N = 8; // Size of input array
    vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8}; // Sample input
    vector<int> output(N, 0); // Output array for OpenCL results
    vector<int> cpu_output(N, 0); // Output array for CPU sequential results

    // Step 1: Print the input array
    printArray(input, "Input Array");

    // Step 2: Perform the scan on the CPU and print the result
    cpu_scan(input, cpu_output, N);
    printArray(cpu_output, "CPU Scan Result");

    // Step 3: Initialize OpenCL
    cluInit();

    // Step 4: Load and compile the OpenCL kernel
    const char *clu_File = SRC_PATH "Parallel_scan/base.cl"; // Path to the kernel file
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "parallel_scan");

    // Step 5: Create OpenCL buffers for input and output
    cl::Buffer inputBuffer(*clu_Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, input.data());
    cl::Buffer outputBuffer(*clu_Context, CL_MEM_READ_WRITE, sizeof(int) * N);

    // Step 6: Set kernel arguments for each stage of the scan
    for (int step = 1; step < N; step *= 2) {
        kernel->setArg(0, inputBuffer);
        kernel->setArg(1, outputBuffer);
        kernel->setArg(2, step);
        kernel->setArg(3, N);  // Size of the array

        // Execute the kernel
        cl::NDRange globalSize(N);  // Number of work items (1 per element)
        clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, cl::NullRange);

        // Copy output back to inputBuffer for the next iteration
        clu_Queue->enqueueCopyBuffer(outputBuffer, inputBuffer, 0, 0, sizeof(int) * N);
    }

    // Step 7: Read the result back from the device to host memory
    clu_Queue->enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(int) * N, output.data());

    // Step 8: Print the OpenCL parallel scan result
    printArray(output, "Parallel Scan Result");

    return 0;
}