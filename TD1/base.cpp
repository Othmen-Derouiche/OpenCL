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

using namespace std;

// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------
// Function to clear the terminal
void clearTerminal() {
    // ANSI escape code to clear the terminal
    cout << "\033[2J\033[1;1H"; // Clear the screen and move cursor to the top left
}

int main(int argc, char **argv)
{
	
	const char *clu_File = SRC_PATH "base.cl";  // path to file containing OpenCL kernel(s) code

	// Initialize OpenCL environment and sets up context and queue
	cluInit();

	// After this call you have access to
	// clu_Context;      <= OpenCL context (pointer)
	// clu_Devices;      <= OpenCL device list (vector)
	// clu_Queue;        <= OpenCL queue (pointer)

	// Load Program
	// Load the OpenCL program from the specified file
	cl::Program *program = cluLoadProgram(clu_File);
	// Load the 'summation' kernel from the program
	cl::Kernel *kernel = cluLoadKernel(program, "game_of_life"); 

	//
	const int WIDTH = 5 ;
	const int HEIGHT = 5 ; 
	const int generations = 5;

	// allocate memory on the compute device 

	const int size = WIDTH * HEIGHT; 

	/*
	// Create buffers for OpenCL to hold input and output data
	cl::Buffer a_buffer(*clu_Context, CL_MEM_READ_ONLY, size * sizeof(int)); 
	cl::Buffer b_buffer(*clu_Context, CL_MEM_READ_ONLY, size * sizeof(int)); 
	cl::Buffer c_buffer(*clu_Context, CL_MEM_WRITE_ONLY, size * sizeof(int)); 
	*/

	// Allocate memory on the compute device
    // Create two buffers for double buffering
    cl::Buffer buffer_current(*clu_Context, CL_MEM_READ_WRITE, WIDTH * HEIGHT * sizeof(int));
    cl::Buffer buffer_next(*clu_Context, CL_MEM_READ_WRITE, WIDTH * HEIGHT * sizeof(int));

	/*
	// Initialize host arrays
	int* a = new int[size]; 
	int* b = new int [size]; 
	for (int i = 0 ; i < size; i++){ 
		a[i] = 0 ; 
		b[i] = i ; 
	}
	*/
	// Initialize host grids with linear mapping
    int* current_grid = new int[WIDTH * HEIGHT](); 
    int* next_grid = new int[WIDTH * HEIGHT]();    

    // Example initialization: a simple "glider" pattern
    // Clear grids
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        current_grid[i] = 0;
        next_grid[i] = 0;
    }
	// Set up a glider pattern
	/*
	current_grid[1 * WIDTH + 2] = 1;
    current_grid[2 * WIDTH + 3] = 1;
    current_grid[3 * WIDTH + 1] = 1;
    current_grid[3 * WIDTH + 2] = 1;
    current_grid[3 * WIDTH + 3] = 1;
	*/

	/*
	current_grid[2 * WIDTH + 1] = 1;
    current_grid[2 * WIDTH + 2] = 1;
    current_grid[2 * WIDTH + 3] = 1;
	*/
	current_grid[1* WIDTH + 2] = 1;
	current_grid[2 * WIDTH + 3] = 1;
	current_grid[3 * WIDTH + 1] = 1;
    current_grid[3 * WIDTH + 2] = 1;
    current_grid[3 * WIDTH + 3] = 1;
	cout << "-------------------------DISPLAY--------------------------"<< endl;
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            std::cout << (current_grid[y * WIDTH + x] ? "█" : "░"); // Full square for alive, empty square for dead
        }
        std::cout << "\n";
    }
    std::cout << "\n";
	/*
	// Write the host arrays to the compute device buffers
	clu_Queue->enqueueWriteBuffer(a_buffer, true, 0, size * sizeof(int), a); 
	clu_Queue->enqueueWriteBuffer(b_buffer, true, 0, size * sizeof(int), b); 
    delete[] a; // Free host memory for a
	delete[] b ;
	*/
	// Write the initial grid to the device
    clu_Queue->enqueueWriteBuffer(buffer_current, true , 0, WIDTH * HEIGHT * sizeof(int), current_grid);

	// Set kernel arguments
	/*
	kernel->setArg(0, a_buffer); 
	kernel->setArg(1, b_buffer); 
	kernel->setArg(2, c_buffer);
	kernel->setArg(3, cl::Local(group_size * sizeof(int))); // Local memory for shared data
	*/

    // Set static kernel arguments that don't change within the loop
    kernel->setArg(2, WIDTH);
    kernel->setArg(3, HEIGHT);

    // Define global and local work sizes
    cl::NDRange global(WIDTH, HEIGHT);
    cl::NDRange local(WIDTH, HEIGHT); // Adjust based on device's max work-group size

	/*
    // Read the results back from the device to host
	int* c = new int[size]; 
	clu_Queue ->enqueueReadBuffer(c_buffer, true, 0, size * sizeof(int), c);
	
	// Print the results
	for (int i = 0 ; i<size ; i++){ 
		std::cout << c[i] << "; "; 
	}
    delete[] c; // Free memory for the result array
	*/
// Simulation loop
    for (int j = 0; j< generations; j++) {

        // Set dynamic kernel arguments
        kernel->setArg(0, buffer_current);
        kernel->setArg(1, buffer_next);

        // Enqueue the kernel
        clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);

        // Wait for kernel execution to finish
        clu_Queue->finish();

        // Swap buffers
        std::swap(buffer_current, buffer_next);

        // Read back the current grid
        clu_Queue->enqueueReadBuffer(buffer_current, CL_TRUE, 0, WIDTH * HEIGHT * sizeof(int), current_grid);

        // Clear the terminal (optional, for better visualization)
        // Uncomment the following lines if you want to clear the terminal each generation
    
        //system("clear");        
		// Clear the terminal using ANSI escape codes
        //clearTerminal();

        // Plot the grid
        std::cout << "Display after " << j + 1 << " Generation :\n";
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                std::cout << (current_grid[y * WIDTH + x] ? "█" : "░"); // Full square for alive, empty square for dead
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        // Optional: Add a short delay for better visualization
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Clean up
    delete[] current_grid;
    delete[] next_grid;
}