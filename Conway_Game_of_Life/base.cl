__kernel void game_of_life(__global const int* current,__global int* next,const int width,const int height,const int enable_periodic_grid){
    
    // Get the x and y coordinates of the current cell
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Ensure that x and y are within bounds
    if (x >= width || y >= height) {
        return;
    }

    // Calculate the indices of the eight neighbors 
    int xm1,xp1,ym1,yp1;
    if (enable_periodic_grid == 1) {
        // Toroidal wrap-around behavior for periodic grid
        xm1 = (x - 1 + width) % width;
        xp1 = (x + 1) % width;
        ym1 = (y - 1 + height) % height;
        yp1 = (y + 1) % height;
    } else {
        // Non-periodic behavior: out-of-bound cells treated as dead
        // If out-of-bounds, mark as -1
        xm1 = (x - 1 < 0) ? -1 : (x - 1);            
        xp1 = (x + 1 >= width) ? -1 : (x + 1);       
        ym1 = (y - 1 < 0) ? -1 : (y - 1);            
        yp1 = (y + 1 >= height) ? -1 : (y + 1);     
    }

    // Calculate the linear indices of the neighbors
    int idx = y * width + x;
    int idx_left = y * width + xm1;
    int idx_right = y * width + xp1;
    int idx_up = ym1 * width + x;
    int idx_down = yp1 * width + x;
    int idx_up_left = ym1 * width + xm1;
    int idx_up_right = ym1 * width + xp1;
    int idx_down_left = yp1 * width + xm1;
    int idx_down_right = yp1 * width + xp1;

    // Count live neighbors
    int live_neighbors =
        current[idx_left] + current[idx_right] +
        current[idx_up] + current[idx_down] +
        current[idx_up_left] + current[idx_up_right] +
        current[idx_down_left] + current[idx_down_right];

    // Apply Conway's Game of Life rules
    if (current[idx] == 1) {
        // Any live cell with two or three live neighbors survives.
        if (live_neighbors == 2 || live_neighbors == 3) {
            next[idx] = 1;
        } else {
            // All other live cells die in the next generation.
            next[idx] = 0;
        }
    } else {
        // Any dead cell with three live neighbors becomes a live cell.
        if (live_neighbors == 3) {
            next[idx] = 1;
        } else {
            // All other dead cells stay dead.
            next[idx] = 0;
        }
    }
}