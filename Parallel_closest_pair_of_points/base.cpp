#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>
#include "clutils.h" // Include your utility functions here

using namespace std;

// Structure to represent a point in 2D space
struct Point {
    float x, y;
};

// Sequential CPU version to calculate the closest pair of points
/*
    Note : this method takes into consideration symetric calculations
*/

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

// Function to generate a vector of random points
vector<Point> generateRandomPoints(int numPoints) {
    vector<Point> points;
    srand(time(0)); 

    for (int i = 0; i < numPoints; ++i) {
        float x = static_cast<float>(rand()) / RAND_MAX * 10.0; // Random float between 0 and 10
        float y = static_cast<float>(rand()) / RAND_MAX * 10.0;

        x = round(x * 100.0) / 100.0;
        y = round(y * 100.0) / 100.0;

        points.push_back({x, y});
    }

    return points;
}
void Print(const vector<Point>& points){
    for (const auto& point : points) {
        std::cout << "Point(" << point.x << ", " << point.y << ")\n";
    }
}
int main() {
    int N = 4 ; // number of points

    //vector<Point> points = {{0.0, 0.0}, {1.0, 2.0}, {4.0, 4.0}, {6.0, 1.0}, {3.0, 5.0}};
    // int N = points.size() :
    vector<Point> points = generateRandomPoints(N);
    vector<float> distances(N * N, numeric_limits<float>::max()); // Result buffer


    // Initialize OpenCL
    cluInit();
    const char *clu_File = SRC_PATH "Parallel_closest_pair_of_points/base.cl"; // Path to the kernel file
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "closest_pair");

    // Prepare OpenCL buffers
    cl::Buffer pointsBuffer(*clu_Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Point) * N, points.data());
    cl::Buffer distancesBuffer(*clu_Context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N);

    // Set kernel arguments
    kernel->setArg(0, pointsBuffer);
    kernel->setArg(1, distancesBuffer);
    kernel->setArg(2, N);

    // Execute the kernel
    cl::NDRange globalSize(N * N);  // Launch num_points * num_points work items
    clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, cl::NullRange);

    // Read back the result from the device
    clu_Queue->enqueueReadBuffer(distancesBuffer, CL_TRUE, 0, sizeof(float) * N * N, distances.data());

    Print(points);
    float cpu_result = cpu_closest_pair(points);
    cout << "CPU Closest Pair Result : "<< cpu_result << endl;

    float gpu_result = *min_element(distances.begin(), distances.end());
    cout << "GPU Closest Pair Result : "<< gpu_result << endl;

    // Compare CPU and GPU results
    //Même si les résultats affichés sont identiques, il peut y avoir des différences mineures en mémoire
    // car la précision interne des opérations en virgule flottante peut varier légèrement entre le CPU et le GPU
    // --> tolerance = 1e-5
    if (!(abs(cpu_result - gpu_result) > 1e-5)) {
        cout << "Results match!" << endl;
    } else {
        cout << "Results do NOT match!" << endl;
    }

    return 0;
}
