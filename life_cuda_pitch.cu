/*
 * The Game of Life
 *
 * a cell is born, if it has exactly three neighbours
 * a cell dies of loneliness, if it has less than two neighbours
 * a cell dies of overcrowding, if it has more than three neighbours
 * a cell survives to the next generation, if it does not die of loneliness
 * or overcrowding
 *
 * In this version, a 2D array of ints is used.  A 1 cell is on, a 0 cell is off.
 * The game plays a number of steps (given by the input), printing to the screen each time.  'x' printed
 * means on, space means off.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned char bool_t;
typedef unsigned char cell_t;

#define TILE_SIZE 8
#define KERNEL_SIZE 3
#define SHARED_MEMORY_SIZE (TILE_SIZE + KERNEL_SIZE - 1)

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

cell_t *allocate_board_flat(int flat_size, int outer_grid_size) {
    cell_t *board = (cell_t *) malloc(sizeof(cell_t) * flat_size);

    for (int i = 0; i < outer_grid_size; ++i) {
        // Fill first row
        board[i] = 0;

        // Fill last row
        board[(outer_grid_size - 1) * outer_grid_size + i] = 0;

        // Fill left column
        board[i * outer_grid_size] = 0;

        // Fill right column
        board[i * outer_grid_size + (outer_grid_size - 1)] = 0;
    }

    return board;
}

__global__ void playKernelSMPitched(const cell_t *d_board, cell_t *d_newboard, size_t pitch, int inner_size, int outer_size) {
    unsigned short bx = blockIdx.x;
    unsigned short by = blockIdx.y;
    unsigned short tx = threadIdx.x;
    unsigned short ty = threadIdx.y;

    // Calculate the row and col for the output array
    unsigned short row_g = by * TILE_SIZE + ty + (KERNEL_SIZE / 2);
    unsigned short col_g = bx * TILE_SIZE + tx + (KERNEL_SIZE / 2);

    __shared__ cell_t neighbors_ds[SHARED_MEMORY_SIZE][SHARED_MEMORY_SIZE];

    unsigned short idx_inner_x = tx + (KERNEL_SIZE / 2);
    unsigned short idx_inner_y = ty + (KERNEL_SIZE / 2);

    unsigned short blockIndex = ty + tx * TILE_SIZE;

    // Using unsigned short reduces the duration of each kernel by ~100 us (~930 us to ~830 us)
    for (unsigned short incr = blockIndex; incr < SHARED_MEMORY_SIZE * SHARED_MEMORY_SIZE; incr += TILE_SIZE * TILE_SIZE) {
        unsigned short ry = incr % SHARED_MEMORY_SIZE;
        unsigned short rx = incr / SHARED_MEMORY_SIZE;

        unsigned short gy = ry + by * TILE_SIZE;
        unsigned short gx = rx + bx * TILE_SIZE;

        // Required to avoid accessing out of bounds
        if (gy < outer_size && gx < outer_size) {
            neighbors_ds[ry][rx] = d_board[gy * pitch + gx];
        }
    }

    // Required so we don't fill the outer padded grid
    if (row_g > inner_size || col_g > inner_size) {
        return;
    }

    // Sync threads now, no need to wait for the threads that exit
    __syncthreads();

    unsigned short a = 0;
    for (unsigned short j = 0; j < KERNEL_SIZE; ++j) {
        for (unsigned short i = 0; i < KERNEL_SIZE; ++i) {
            a += neighbors_ds[j + idx_inner_y - (KERNEL_SIZE / 2)][i + idx_inner_x - (KERNEL_SIZE / 2)];
        }
    }
    a -= neighbors_ds[idx_inner_y][idx_inner_x];

    if (a == 2)
        d_newboard[row_g * pitch + col_g] = neighbors_ds[idx_inner_y][idx_inner_x];
    if (a == 3)
        d_newboard[row_g * pitch + col_g] = 1;
    if (a < 2)
        d_newboard[row_g * pitch + col_g] = 0;
    if (a > 3)
        d_newboard[row_g * pitch + col_g] = 0;
}

/* print the life board */
void print_flat(cell_t *board, int inner_size, int outer_size) {
    int i, j;
    /* for each row */
    for (j = 0; j < inner_size; j++) {
        /* print each column position... */
        for (i = 0; i < inner_size; i++)
            printf("%c", board[(j + (KERNEL_SIZE / 2)) * outer_size + (i + (KERNEL_SIZE / 2))] ? 'x' : ' ');
        /* followed by a carriage return */
        printf("\n");
    }
}

/* read a file into the life board */
void read_file_flat(FILE *f, cell_t *board, int inner_size, int outer_size) {
    int i, j;
    size_t len;
    char *s = (char *) malloc(inner_size + 10);

    for (j = 0; j < inner_size; j++) {
        /* get a string */
        fgets(s, inner_size + 10, f);
        len = strlen(s) - 1;

        /* copy the string to the life board */
        for (i = 0; i < inner_size; i++) {
            board[(j + (KERNEL_SIZE / 2)) * outer_size + (i + (KERNEL_SIZE / 2))] = i < len ? s[i] == 'x' : 0;
        }
    }
}

int main(int argc, char *argv[]) {
    // Host variables
    int size, flat_size, steps, i, grid_size, outer_grid_size;
    FILE *f_in;
    cell_t *h_prev;
    bool_t writeOutput = 1, evenSteps;
    size_t pitch;

    // Device variables
    cell_t *d_prev, *d_next;

    f_in = stdin;

    // Read the input file and write its content in the host array
    fscanf(f_in, "%d %d", &size, &steps);

    // Create a border around the grid to avoid dealing with boundary conditions
    outer_grid_size = size + (2 * (KERNEL_SIZE / 2));
    flat_size = outer_grid_size * outer_grid_size;
    evenSteps = steps % 2 == 0;

    h_prev = allocate_board_flat(flat_size, outer_grid_size);
    read_file_flat(f_in, h_prev, size, outer_grid_size);
    fclose(f_in);

    grid_size = int(ceil((float) size / TILE_SIZE));

    dim3 dimGrid(grid_size, grid_size, 1);

    // In our case, a TILE_SIZE of 8 gives the best results, with 16 and 32 being slightly slower
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    // Allocate device arrays
    gpuErrchk(cudaMallocPitch((void **) &d_prev, &pitch, outer_grid_size * sizeof(cell_t), outer_grid_size));
    gpuErrchk(cudaMallocPitch((void **) &d_next, &pitch, outer_grid_size * sizeof(cell_t), outer_grid_size));

    // Copy the data from the host array to the device array
    gpuErrchk(cudaMemcpy2D(d_prev, pitch,
                 h_prev, outer_grid_size * sizeof(cell_t),
                 outer_grid_size * sizeof(cell_t), outer_grid_size,
                 cudaMemcpyHostToDevice));

    for (i = 0; i < int(ceil((float) steps / 2)); i++) {
        //  printf("Step: %d\n", 2 * i);

        // Instead of using cudaMemcpy and a buffer or swapping pointers,
        // run the same kernel with the variables inverted
        playKernelSMPitched<<<dimGrid, dimBlock>>>(d_prev, d_next, pitch, size, outer_grid_size);

        if (evenSteps || (2 * i + 1) < steps) {
            // printf("Step: %d\n", 2 * i + 1);
            playKernelSMPitched<<<dimGrid, dimBlock>>>(d_next, d_prev, pitch, size, outer_grid_size);
        }
    }

    // Copy data back from the device array to the host array
    gpuErrchk(cudaMemcpy2D(h_prev, outer_grid_size * sizeof(cell_t),
                 evenSteps ? d_prev : d_next, pitch,
                 outer_grid_size * sizeof(cell_t), outer_grid_size,
                 cudaMemcpyDeviceToHost));

    // Deallocate device arrays
    gpuErrchk(cudaFree(d_next));
    gpuErrchk(cudaFree(d_prev));

    if (writeOutput) {
        print_flat(h_prev, size, outer_grid_size);
    }

    free(h_prev);

    return EXIT_SUCCESS;
}
