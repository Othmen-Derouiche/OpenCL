__kernel void parallel_scan(__global float* input, __global float* output, const int step, const int N) {
    int gid = get_global_id(0);

    if (gid >= step && gid < N) {
        output[gid] = input[gid] + input[gid - step];  // Current element + previous element at "step" distance
    } else {
        output[gid] = input[gid];  // No change if we're at the beginning of the array
    }
}