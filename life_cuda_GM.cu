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

cell_t *allocate_board_flat(int flat_size) {
    cell_t *board = (cell_t *) malloc(sizeof(cell_t) * flat_size);

    return board;
}

__device__ int adjacent_to(cell_t *d_board, int size, int i, int j) {
    int k, l, count = 0;

    int sk = (i > 0) ? i - 1 : i;
    int ek = (i + 1 < size) ? i + 1 : i;
    int sl = (j > 0) ? j - 1 : j;
    int el = (j + 1 < size) ? j + 1 : j;

    for (k = sk; k <= ek; k++)
        for (l = sl; l <= el; l++)
            count += d_board[l * size + k];
    count -= d_board[j * size + i];

    return count;
}

__global__ void playKernel(cell_t *d_board, cell_t *d_newboard, int size) {
    int a;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int d_j = by * blockDim.y + ty;
    int d_i = bx * blockDim.x + tx;

    if (d_j < size && d_i < size) {
        a = adjacent_to(d_board, size, d_i, d_j);

        if (a == 2)
            d_newboard[d_j * size + d_i] = d_board[d_j * size + d_i];
        if (a == 3)
            d_newboard[d_j * size + d_i] = 1;
        if (a < 2)
            d_newboard[d_j * size + d_i] = 0;
        if (a > 3)
            d_newboard[d_j * size + d_i] = 0;
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
void read_file_flat(FILE *f, cell_t *board, int size) {
    int i, j;
    char *s = (char *) malloc(size + 10);

    for (j = 0; j < size; j++) {
        /* get a string */
        fgets(s, size + 10, f);
        /* copy the string to the life board */
        for (i = 0; i < size; i++) {
            board[j * size + i] = s[i] == 'x';
        }
    }
}

void write_file_flat(FILE *f, cell_t *board, int size) {
    int i, j;
    char *s = (char *) malloc(size + 10);

    for (j = 0; j < size; j++) {
        /* print each column position... */
        for (i = 0; i < size; i++)
            fprintf(f, "%c", board[j * size + i] ? 'x' : ' ');
        /* followed by a carriage return */
        fprintf(f, "\n");
    }
}

int main(int argc, char *argv[]) {
    // Host variables
    int size, flat_size, steps, i, grid_size;
    FILE *f_in;
    cell_t *h_prev;
    bool_t writeOutput = 1, evenSteps;

    // Device variables
    cell_t *d_prev, *d_next;

    f_in = stdin;

    // Read the input file and write its content in the host array
    fscanf(f_in, "%d %d", &size, &steps);

    flat_size = size * size;
    evenSteps = steps % 2 == 0;

    h_prev = allocate_board_flat(flat_size);
    read_file_flat(f_in, h_prev, size);
    fclose(f_in);

    grid_size = int(ceil((float) size / TILE_SIZE));

    dim3 dimGrid(grid_size, grid_size, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    // Allocate device arrays
    gpuErrchk(cudaMalloc((void **) &d_prev, flat_size * sizeof(cell_t)));
    gpuErrchk(cudaMalloc((void **) &d_next, flat_size * sizeof(cell_t)));

    // Copy the data from the host array to the device array
    gpuErrchk(cudaMemcpy(d_prev, h_prev, flat_size * sizeof(cell_t), cudaMemcpyHostToDevice));

    for (i = 0; i < int(ceil((float) steps / 2)); i++) {
        //  printf("Step: %d\n", 2 * i);

        // Instead of using cudaMemcpy and a buffer or swapping pointers,
        // run the same kernel with the variables inverted
        playKernel<<<dimGrid, dimBlock>>>(d_prev, d_next, size);

        if (evenSteps || (2 * i + 1) < steps) {
            // printf("Step: %d\n", 2 * i + 1);
            playKernel<<<dimGrid, dimBlock>>>(d_next, d_prev, size);
        }
    }

    // Copy data back from the device array to the host array
    if (!evenSteps) {
        gpuErrchk(cudaMemcpy(h_prev, d_next, flat_size * sizeof(cell_t), cudaMemcpyDeviceToHost));
    } else {
        gpuErrchk(cudaMemcpy(h_prev, d_prev, flat_size * sizeof(cell_t), cudaMemcpyDeviceToHost));
    }

    // Deallocate device arrays
    gpuErrchk(cudaFree(d_next));
    gpuErrchk(cudaFree(d_prev));

    if (writeOutput) {
        print_flat(h_prev, size);
    }

    free(h_prev);

    return EXIT_SUCCESS;
}
