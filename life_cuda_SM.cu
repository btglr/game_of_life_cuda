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

/**
 * Allocates the board as a 1D array instead of the previous 2D array
 * @param flat_size
 * @param outer_grid_size
 * @return An allocated board of the given flat_size
 */
cell_t *allocate_board_flat(unsigned int flat_size, int outer_grid_size) {
    cell_t *board = (cell_t *) malloc(sizeof(cell_t) * flat_size);

    // Since there is no dependency between the indices we can collapse the loop
    // Only if the grid is large enough
#pragma omp parallel for collapse(2) if(outer_grid_size >= 5000)
    for (int i = 0; i < outer_grid_size; ++i) {
        for (int k = 0; k < KERNEL_SIZE / 2; ++k) {
            // Fill first rows
            board[k * outer_grid_size + i] = 0;

            // Fill last rows
            board[(outer_grid_size - (k + 1)) * outer_grid_size + i] = 0;

            // Fill left columns
            board[i * outer_grid_size + k] = 0;

            // Fill right columns
            board[i * outer_grid_size + (outer_grid_size - (k + 1))] = 0;
        }
    }

    return board;
}

__global__ void playKernelSM(const cell_t *d_board, cell_t *d_newboard, int inner_size, int outer_size) {
    int a = 0;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate the row and col for the output array
    int row_g = by * TILE_SIZE + ty + 1;
    int col_g = bx * TILE_SIZE + tx + 1;

    int previous_row_g = row_g - 1;
    int previous_col_g = col_g - 1;
    int next_row_g = row_g + 1;
    int next_col_g = col_g + 1;

    // Add 1 on each side for each dimension for the edges
    __shared__ cell_t neighbors_ds[TILE_SIZE + 2][TILE_SIZE + 2];

    int idx_inner_x = tx + 1;
    int idx_inner_y = ty + 1;

    if (row_g <= inner_size && col_g <= inner_size) {
        // If ty is 0, load the value from the row above
        if (ty == 0) {
            // Previous row, same column
            neighbors_ds[idx_inner_y - 1][idx_inner_x] = d_board[previous_row_g * outer_size + col_g];

            // Load top left corner
            if (tx == 0) {
                neighbors_ds[idx_inner_y - 1][idx_inner_x - 1] = d_board[previous_row_g * outer_size + previous_col_g];
            }
                // Load top right corner
            else if (tx == (TILE_SIZE - 1)) {
                neighbors_ds[idx_inner_y - 1][idx_inner_x + 1] = d_board[previous_row_g * outer_size + next_col_g];
            }
        } else if (ty == (TILE_SIZE - 1)) {
            // Next row, same column
            neighbors_ds[idx_inner_y + 1][idx_inner_x] = d_board[next_row_g * outer_size + col_g];

            // Load bottom left corner
            if (tx == 0) {
                neighbors_ds[idx_inner_y + 1][idx_inner_x - 1] = d_board[next_row_g * outer_size + previous_col_g];
            }
                // Load bottom right corner
            else if (tx == (TILE_SIZE - 1)) {
                neighbors_ds[idx_inner_y + 1][idx_inner_x + 1] = d_board[next_row_g * outer_size + next_col_g];
            }
        }

        // If tx is 0, load the value from the column to the left
        if (tx == 0) {
            // Same row, previous column
            neighbors_ds[idx_inner_y][idx_inner_x - 1] = d_board[row_g * outer_size + previous_col_g];
        } else if (tx == (TILE_SIZE - 1)) {
            // Same row, next column
            neighbors_ds[idx_inner_y][idx_inner_x + 1] = d_board[row_g * outer_size + next_col_g];
        }

        // Load normally
        if (idx_inner_x >= 1 && idx_inner_y >= 1 && idx_inner_x <= TILE_SIZE && idx_inner_y <= TILE_SIZE) {
            neighbors_ds[idx_inner_y][idx_inner_x] = d_board[row_g * outer_size + col_g];
        }
    } else {
        return;
        // neighbors_ds[idx_inner_y][idx_inner_x] = 0;
    }

    __syncthreads();

    if (idx_inner_y <= TILE_SIZE && idx_inner_x <= TILE_SIZE) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                a += neighbors_ds[j + idx_inner_y - 1][i + idx_inner_x - 1];
            }
        }
        a -= neighbors_ds[idx_inner_y][idx_inner_x];

        if (a == 2)
            d_newboard[row_g * outer_size + col_g] = neighbors_ds[idx_inner_y][idx_inner_x];
        if (a == 3)
            d_newboard[row_g * outer_size + col_g] = 1;
        if (a < 2)
            d_newboard[row_g * outer_size + col_g] = 0;
        if (a > 3)
            d_newboard[row_g * outer_size + col_g] = 0;
    }
}

/**
 * Processes one step of the game of life
 * @param d_board The original board
 * @param d_newboard The new board
 * @param inner_size The inner grid size
 * @param outer_size The padded outer grid size
 */
__global__ void playKernelSMDynamic(const cell_t *d_board, cell_t *d_newboard, int inner_size, int outer_size) {
    unsigned short bx = blockIdx.x;
    unsigned short by = blockIdx.y;
    unsigned short tx = threadIdx.x;
    unsigned short ty = threadIdx.y;

    // Calculate the row and col for the output array
    unsigned short row = by * TILE_SIZE + ty + (KERNEL_SIZE / 2);
    unsigned short col = bx * TILE_SIZE + tx + (KERNEL_SIZE / 2);

    __shared__ cell_t ds_neighbors[SHARED_MEMORY_SIZE][SHARED_MEMORY_SIZE];

    // These indices are the inner grid's indices
    // For example, with a 5000x5000 padded on each side, we get a 5002x5002 grid
    // To access the index (0, 0) of the inner grid, we must access the index (1, 1) of the padded grid
    unsigned short idx_inner_x = tx + (KERNEL_SIZE / 2);
    unsigned short idx_inner_y = ty + (KERNEL_SIZE / 2);

    unsigned short blockIndex = ty + tx * TILE_SIZE;

    // Using unsigned short reduces the duration of each kernel by ~100 us (~930 us to ~830 us)
    // Here, each thread is responsible for loading between one and two values into the shared memory array
    for (unsigned short incr = blockIndex; incr < SHARED_MEMORY_SIZE * SHARED_MEMORY_SIZE; incr += TILE_SIZE * TILE_SIZE) {
        unsigned short ry = incr % SHARED_MEMORY_SIZE;
        unsigned short rx = incr / SHARED_MEMORY_SIZE;

        unsigned short gy = ry + by * TILE_SIZE;
        unsigned short gx = rx + bx * TILE_SIZE;

        // Required to avoid accessing out of bounds
        if (gy < outer_size && gx < outer_size) {
            ds_neighbors[ry][rx] = d_board[gy * outer_size + gx];
        }
    }

    // Required so we don't fill the outer padded grid
    if (row > inner_size || col > inner_size) {
        return;
    }

    // Sync threads now, no need to wait for the threads that exit
    __syncthreads();

    unsigned short a = 0;

    // Instead of using the serial function that had multiple conditions checking for boundaries, simply loop over all
    // neighbors as the shared memory array is padded on each side too
    for (unsigned short j = 0; j < KERNEL_SIZE; ++j) {
        for (unsigned short i = 0; i < KERNEL_SIZE; ++i) {
            a += ds_neighbors[j + idx_inner_y - (KERNEL_SIZE / 2)][i + idx_inner_x - (KERNEL_SIZE / 2)];
        }
    }
    a -= ds_neighbors[idx_inner_y][idx_inner_x];

    if (a == 2)
        d_newboard[row * outer_size + col] = ds_neighbors[idx_inner_y][idx_inner_x];
    if (a == 3)
        d_newboard[row * outer_size + col] = 1;
    if (a < 2)
        d_newboard[row * outer_size + col] = 0;
    if (a > 3)
        d_newboard[row * outer_size + col] = 0;
}

/**
 * Print the game of the life board from a flat 1D array
 * @param board The board
 * @param inner_size The inner grid size
 * @param outer_size The padded outer grid size
 */
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

/**
 * Read the game of the life board into a flat 1D array from a file
 * @param f The file to read
 * @param board The allocated 1D array
 * @param inner_size The inner grid size
 * @param outer_size The padded outer grid size
 */
void read_file_flat(FILE *f, cell_t *board, int inner_size, int outer_size) {
    int i, j;
    size_t len;
    char *s = (char *) malloc(inner_size + 10);

    for (j = 0; j < inner_size; j++) {
        /* get a string */
        fgets(s, inner_size + 10, f);

        // Get the length of the line that was just read since fgets adds a '\0' at the end of a line
        len = strlen(s) - 1;

        /* copy the string to the life board */
        for (i = 0; i < inner_size; i++) {
            // If i is less than the line's length, read the value at that index, otherwise fill with a blank (0)
            // The serial version was accessing lines out of bounds due to the first line of judge.in not being
            // as long as the grid size expects it to be
            // Those accesses sometimes added 'x's where they shouldn't have been if the accessed memory wasn't empty,
            // causing the result to be wrong with a small number of steps

            // Since the grid is padded on each side to avoid boundary conditions, only fill the 'inner' part
            board[(j + (KERNEL_SIZE / 2)) * outer_size + (i + (KERNEL_SIZE / 2))] = i < len ? s[i] == 'x' : 0;
        }
    }
}

int main() {
    // Host variables
    unsigned short size, steps, i, grid_size, outer_grid_size;
    unsigned int flat_size;
    FILE *f_in;
    cell_t *h_prev;
    bool_t writeOutput = 1, evenSteps;

    // Device variables
    cell_t *d_prev, *d_next;

    f_in = stdin;

    // Read the input file and write its content in the host array
    fscanf(f_in, "%d %d", &size, &steps);

    // Create a border around the grid to avoid dealing with boundary conditions
    // Assuming a KERNEL_SIZE of 3, a 5000x5000 grid will be padded with 1 extra row/column on each side, resulting
    // in a 5002x5002 grid
    outer_grid_size = size + (2 * (KERNEL_SIZE / 2));
    flat_size = outer_grid_size * outer_grid_size;
    evenSteps = steps % 2 == 0;

    // Allocate a 'flat' array instead of the previous 2D array so the memory is contiguous for faster accesses
    h_prev = allocate_board_flat(flat_size, outer_grid_size);
    read_file_flat(f_in, h_prev, size, outer_grid_size);
    fclose(f_in);

    // Assuming we only have square grids, the grid size is the same for both dimensions
    // Here, we round up the result to make sure even a grid size that's not a multiple of 32 will be fully processed
    // at the cost of having more blocks than necessary
    grid_size = int(ceilf((float) size / TILE_SIZE));

    dim3 dimGrid(grid_size, grid_size, 1);

    // In our case, a TILE_SIZE of 8 gives the best results, with 16 and 32 being slightly slower (+ ~20 ms)
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    // Allocate device arrays
    cudaMalloc((void **) &d_prev, flat_size * sizeof(cell_t));
    cudaMalloc((void **) &d_next, flat_size * sizeof(cell_t));

    // Copy the data from the host array to the device array
    cudaMemcpy(d_prev, h_prev, flat_size * sizeof(cell_t), cudaMemcpyHostToDevice);

    for (i = 0; i < int(ceilf((float) steps / 2)); i++) {
//        printf("Step: %d\n", 2 * i);

        // Instead of using cudaMemcpy and a buffer or swapping pointers, run the same kernel with the variables inverted
        // The data remains on the GPU at all times
        playKernelSMDynamic<<<dimGrid, dimBlock>>>(d_prev, d_next, size, outer_grid_size);

        // Condition to make sure the second kernel is only executed if necessary
        // For example, 500 steps can be divided by 2 (judge.in)
        // But 11 steps can't (life.in), thus ceilf(steps / 2) returns 12 which would result in an extra step
        if (evenSteps || (2 * i + 1) < steps) {
//            printf("Step: %d\n", 2 * i + 1);
            playKernelSMDynamic<<<dimGrid, dimBlock>>>(d_next, d_prev, size, outer_grid_size);
        }
    }

    // Copy data back from the device array to the host array
    if (!evenSteps) {
        cudaMemcpy(h_prev, d_next, flat_size * sizeof(cell_t), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_prev, d_prev, flat_size * sizeof(cell_t), cudaMemcpyDeviceToHost);
    }

    // Deallocate device arrays
    cudaFree(d_next);
    cudaFree(d_prev);

    if (writeOutput) {
        print_flat(h_prev, size, outer_grid_size);
    }

    free(h_prev);

    return EXIT_SUCCESS;
}
