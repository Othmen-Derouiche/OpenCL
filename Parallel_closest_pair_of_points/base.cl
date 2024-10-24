__kernel void closest_pair(__global const float2* points, __global float* distances, const int n) {
    int gid = get_global_id(0);

    // Get the two points indices from the global ID 
    int i = gid / n;
    int j = gid % n;

    // Only calculate distance for distinct points (i != j)
    // avoid calculating the distance of the point and itseld
    if (i < j && i < n && j < n) {
        float2 p1 = points[i];
        float2 p2 = points[j];

        // Calculate the Euclidean distance between points i and j
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dist = sqrt(dx * dx + dy * dy);

        // Store the distance in the corresponding buffer
        distances[gid] = dist;
    } else {
        distances[gid] = FLT_MAX; // Set to a large value for invalid comparisons
    }
}