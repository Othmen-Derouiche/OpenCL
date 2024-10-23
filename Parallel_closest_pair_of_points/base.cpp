#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <CL/cl.hpp>
#include "clutils.h" // Include your utility functions here

using namespace std;

// Structure to represent a point in 2D space
struct Point {
    float x, y;
};

// Sequential CPU version to calculate the closest pair of points
float cpu_closest_pair(const vector<Point>& points) {
    int num_points = points.size();
    float min_distance = numeric_limits<float>::max();

    // Iterate through all pairs of points
    for (int i = 0; i < num_points; ++i) {
        for (int j = i + 1; j < num_points; ++j) {
            float dx = points[i].x - points[j].x;
            float dy = points[i].y - points[j].y;
            float distance = sqrt(dx * dx + dy * dy);
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
    }

    return min_distance;
}

// Function to print the result
void print_result(const string& label, float result) {
    cout << label << ": " << result << endl;
}

int main() {
    // Step 1: Initialize the points
    vector<Point> points = {{0.0, 0.0}, {1.0, 2.0}, {4.0, 4.0}, {6.0, 1.0}, {3.0, 5.0}};
    int num_points = points.size();

    // Step 2: Perform the CPU closest pair computation
    float cpu_result = cpu_closest_pair(points);
    print_result("CPU Closest Pair Result", cpu_result);

    // Step 3: Initialize OpenCL
    cluInit();
    const char *clu_File = SRC_PATH "Parallel_closest_pair_of_points/base.cl"; // Path to the kernel file
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "closest_pair");

    // Step 4: Prepare OpenCL buffers
    vector<float> distances(num_points * num_points, numeric_limits<float>::max()); // Result buffer
    cl::Buffer pointsBuffer(*clu_Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Point) * num_points, points.data());
    cl::Buffer distancesBuffer(*clu_Context, CL_MEM_WRITE_ONLY, sizeof(float) * num_points * num_points);

    // Step 5: Set kernel arguments
    kernel->setArg(0, pointsBuffer);
    kernel->setArg(1, distancesBuffer);
    kernel->setArg(2, num_points);

    // Step 6: Execute the kernel
    cl::NDRange globalSize(num_points * num_points);  // Launch num_points * num_points work items
    clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, cl::NullRange);

    // Step 7: Read back the result from the device
    clu_Queue->enqueueReadBuffer(distancesBuffer, CL_TRUE, 0, sizeof(float) * num_points * num_points, distances.data());

    // Step 8: Find the minimum distance in the result array
    float gpu_result = *min_element(distances.begin(), distances.end());
    print_result("GPU Closest Pair Result", gpu_result);

    // Step 9: Compare CPU and GPU results
    if (abs(cpu_result - gpu_result) < 1e-6) {
        cout << "Results match!" << endl;
    } else {
        cout << "Results do NOT match!" << endl;
    }

    return 0;
}
