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

#define TILE_SIZE 32

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

    // Add 1 on each side for the edges
    __shared__ cell_t ds[TILE_SIZE + 2][TILE_SIZE + 2];

    int idx_inner_x = tx + 1;
    int idx_inner_y = ty + 1;

    if (row_g <= inner_size && col_g <= inner_size) {
        // If ty is 0 (top edge), load an extra value from the row above
        // If ty is 31 (bottom edge), load an extra value from the row below
        if (ty == 0) {
            // Previous row, same column
            ds[idx_inner_y - 1][idx_inner_x] = d_board[previous_row_g * pitch + col_g];

            // Load top left/right corner
            if (tx == 0) ds[idx_inner_y - 1][idx_inner_x - 1] = d_board[previous_row_g * pitch + previous_col_g];
            else if (tx == (TILE_SIZE - 1))
                ds[idx_inner_y - 1][idx_inner_x + 1] = d_board[previous_row_g * pitch + next_col_g];
        } else if (ty == (TILE_SIZE - 1)) {
            // Next row, same column
            ds[idx_inner_y + 1][idx_inner_x] = d_board[next_row_g * pitch + col_g];

            // Load bottom left/right corner
            if (tx == 0) ds[idx_inner_y + 1][idx_inner_x - 1] = d_board[next_row_g * pitch + previous_col_g];
            else if (tx == (TILE_SIZE - 1))
                ds[idx_inner_y + 1][idx_inner_x + 1] = d_board[next_row_g * pitch + next_col_g];
        }

        // If tx is 0 (left edge), load an extra value from the column to the left
        // If tx is 31 (right edge), load an extra value from the column to the right
        if (tx == 0) {
            // Same row, previous column
            ds[idx_inner_y][idx_inner_x - 1] = d_board[row_g * pitch + previous_col_g];
        } else if (tx == (TILE_SIZE - 1)) {
            // Same row, next column
            ds[idx_inner_y][idx_inner_x + 1] = d_board[row_g * pitch + next_col_g];
        }

        // Each thread that's not directly on the edges loads its own value
        if (idx_inner_x >= 1 && idx_inner_y >= 1 && idx_inner_x <= TILE_SIZE && idx_inner_y <= TILE_SIZE) {
            ds[idx_inner_y][idx_inner_x] = d_board[row_g * pitch + col_g];
        }
    } else {
        return;
    }

    __syncthreads();

    if (idx_inner_y <= TILE_SIZE && idx_inner_x <= TILE_SIZE) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                a += ds[j + idx_inner_y - 1][i + idx_inner_x - 1];
            }
        }
        a -= ds[idx_inner_y][idx_inner_x];

        if (a == 2)
            d_newboard[row_g * pitch + col_g] = ds[idx_inner_y][idx_inner_x];
        if (a == 3)
            d_newboard[row_g * pitch + col_g] = 1;
        if (a < 2)
            d_newboard[row_g * pitch + col_g] = 0;
        if (a > 3)
            d_newboard[row_g * pitch + col_g] = 0;
    }
}

/* print the life board */
void print_flat(cell_t *board, int size) {
    int i, j;
    /* for each row */
    for (j = 0; j < size; j++) {
        /* print each column position... */
        for (i = 0; i < size; i++)
            printf("%c", board[j * size + i] ? 'x' : ' ');
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
            board[(j + 1) * outer_size + (i + 1)] = i < len ? s[i] == 'x' : 0;
        }
    }
}

void write_file_flat(FILE *f, cell_t *board, int inner_size, int outer_size) {
    int i, j;
    char *s = (char *) malloc(inner_size + 10);

    for (j = 0; j < inner_size; j++) {
        /* print each column position... */
        for (i = 0; i < inner_size; i++)
            fprintf(f, "%c", board[(j + 1) * outer_size + (i + 1)] ? 'x' : ' ');
        /* followed by a carriage return */
        fprintf(f, "\n");
    }
}

void usage() {
    printf("Usage: ./life input_file [output_file]\n");
    printf("input_file: path to the input file\n");
    printf("output_file: path to the output file\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage();
        return EXIT_FAILURE;
    }

    // Host variables
    int size, flat_size, steps, i, grid_size, outer_grid_size;
    FILE *f_in, *f_out = NULL;
    cell_t *h_prev;
    bool_t writeOutput = 0, evenSteps;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    size_t pitch;

    // Device variables
    cell_t *d_prev, *d_next;

    // Prepare the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Open files
    printf("Provided input file: %s\n", argv[1]);
    f_in = fopen(argv[1], "r");

    if (argc >= 3) {
        printf("Provided output file: %s\n", argv[2]);
        f_out = fopen(argv[2], "w+");
        writeOutput = 1;
    }

    // Read the input file and write its content in the host array
    fscanf(f_in, "%d %d", &size, &steps);

    outer_grid_size = size + 2;
    flat_size = outer_grid_size * outer_grid_size;
    evenSteps = steps % 2 == 0;

    h_prev = allocate_board_flat(flat_size, outer_grid_size);
    read_file_flat(f_in, h_prev, size, outer_grid_size);
    fclose(f_in);

//    for (int bla = 0; bla < flat_size; ++bla) {
//        printf("%d", h_prev[bla]);
//    }

    grid_size = int(ceil((float) size / TILE_SIZE));

    dim3 dimGrid(grid_size, grid_size, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    // Allocate device arrays
    cudaMallocPitch((void **) &d_prev, &pitch, outer_grid_size * sizeof(cell_t), outer_grid_size);
    cudaMallocPitch((void **) &d_next, &pitch, outer_grid_size * sizeof(cell_t), outer_grid_size);

    printf("Pitch: %lu\n", pitch);

    // Copy the data from the host array to the device array

    cudaMemcpy2D(d_prev, pitch,
                 h_prev, outer_grid_size * sizeof(cell_t),
                 outer_grid_size * sizeof(cell_t), outer_grid_size,
                 cudaMemcpyHostToDevice);
//    cudaMemcpy(d_prev, h_prev, flat_size * sizeof(cell_t), cudaMemcpyHostToDevice);

    cudaEventRecord(start);

//    steps = 1;
//    evenSteps = false;

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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Game of life (%d steps) done in %lf seconds\n", steps, milliseconds / 1000);

    // Copy data back from the device array to the host array
    if (!evenSteps) {
        cudaMemcpy2D(h_prev, outer_grid_size * sizeof(cell_t),
                     d_next, pitch,
                     outer_grid_size * sizeof(cell_t), outer_grid_size,
                     cudaMemcpyDeviceToHost);

//        cudaMemcpy(h_prev, d_next, flat_size * sizeof(cell_t), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy2D(h_prev, outer_grid_size * sizeof(cell_t),
                     d_prev, pitch,
                     outer_grid_size * sizeof(cell_t), outer_grid_size,
                     cudaMemcpyDeviceToHost);

//        cudaMemcpy(h_prev, d_prev, flat_size * sizeof(cell_t), cudaMemcpyDeviceToHost);
    }

    // Deallocate device arrays
    cudaFree(d_next);
    cudaFree(d_prev);

    if (writeOutput) {
        printf("Writing output file...\n");
        write_file_flat(f_out, h_prev, size, outer_grid_size);
        fclose(f_out);
    }

    free(h_prev);

    return EXIT_SUCCESS;
}
