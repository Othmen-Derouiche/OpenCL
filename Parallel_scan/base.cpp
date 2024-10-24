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
#define ARRAY_SIZE 20  // Size of the input array
// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------

// Sequential CPU version of the scan algorithm 
void cpu_scan(const vector<float>& input, vector<float>& output) {
    output[0] = input[0];
    for (int i = 1; i < input.size(); ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

// Function to print an array
void printArray(const vector<float>& arr, const string& label) {
    cout << label << ": ";
    for (const auto& val : arr) {
        cout << val << " ";
    }
    cout << endl;
}
// Verify vectors equality
bool verif(const vector<float>& vec1, const vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (abs(vec1[i] - vec2[i]) > 1e-3) {
            // une tolerance de 1e-5 ne marche pas dans tous les cas --> besoin d'augmenter la tolérance d'ou 1e-3 
            return false;
        }
    }
    return true;
}
vector<float> generateRandomArray(int size) {
    vector<float> array;
    srand(time(0)); // Seed for random number generation

    for (int i = 0; i < size; ++i) {
        float val = static_cast<float>(rand()) / RAND_MAX * 10.0f; // Random float between 0 and 10
        array.push_back(val);
    }

    return array;
}
int main(int argc, char **argv) {
    //vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8}; // Sample input
    vector<float> input = generateRandomArray(ARRAY_SIZE);
    vector<float> gpu_output(ARRAY_SIZE, 0); // Output array for OpenCL results
    vector<float> cpu_output(ARRAY_SIZE, 0); // Output array for CPU sequential results

    // Step 1: Initialize OpenCL
    cluInit();

    // Step 2: Load and compile the OpenCL kernel
    const char *clu_File = SRC_PATH "Parallel_scan/base.cl"; // Path to the kernel file
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "parallel_scan");

    // Step 3: Create OpenCL buffers for input and output
    cl::Buffer inputBuffer(*clu_Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, input.data());
    cl::Buffer outputBuffer(*clu_Context, CL_MEM_READ_WRITE, sizeof(int) * ARRAY_SIZE);

    // Step 4: Set kernel arguments for each stage of the scan
    for (int d = 1; d < ARRAY_SIZE; d = d << 1) {
        kernel->setArg(0, inputBuffer);
        kernel->setArg(1, outputBuffer);
        kernel->setArg(2, d);
        kernel->setArg(3, ARRAY_SIZE);  // Size of the array

        // Execute the kernel
        cl::NDRange globalSize(ARRAY_SIZE);  // Number of work items (1 per element)
        clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, cl::NullRange);

        // Copy output back to inputBuffer for the next iteration
        clu_Queue->enqueueCopyBuffer(outputBuffer, inputBuffer, 0, 0, sizeof(int) * ARRAY_SIZE);
    }

    // Step 5: Read the result back from the device to host memory
    clu_Queue->enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, gpu_output.data());

    // Step 6: Print the scan results
    printArray(input, "Input Array");
    cpu_scan(input, cpu_output);
    printArray(cpu_output, "CPU Scan Result");
    printArray(gpu_output, "Parallel Scan Result using GPU");

    bool v = verif(cpu_output,gpu_output);
    if (v) cout << "Same result between CPU & GPU" << endl;
        else cout << "Error in GPU calculation" << endl;


    return 0;
}