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

typedef unsigned char cell_t;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

cell_t *allocate_board_flat(int flat_size) {
    cell_t *board = (cell_t *) malloc(sizeof(cell_t) * flat_size);

    return board;
}

cell_t **allocate_board(int size) {
    cell_t **board = (cell_t **) malloc(sizeof(cell_t *) * size);
    int i;
    for (i = 0; i < size; i++)
        board[i] = (cell_t *) malloc(sizeof(cell_t) * size);
    return board;
}

void free_board(cell_t **board, int size) {
    int i;
    for (i = 0; i < size; i++)
        free(board[i]);
    free(board);
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


/* return the number of on cells adjacent to the i,j cell */
/* int adjacent_to(cell_t **board, int size, int i, int j) {
    int k, l, count = 0;

    int sk = (i > 0) ? i - 1 : i;
    int ek = (i + 1 < size) ? i + 1 : i;
    int sl = (j > 0) ? j - 1 : j;
    int el = (j + 1 < size) ? j + 1 : j;

    for (k = sk; k <= ek; k++)
        for (l = sl; l <= el; l++)
            count += board[k][l];
    count -= board[i][j];

    return count;
} */

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
        if (a == 2) d_newboard[d_j * size + d_i] = d_board[d_j * size + d_i];
        if (a == 3) d_newboard[d_j * size + d_i] = 1;
        if (a < 2) d_newboard[d_j * size + d_i] = 0;
        if (a > 3) d_newboard[d_j * size + d_i] = 0;
    }
}

/* void play(cell_t **board, cell_t **newboard, int size) {
    int i, j, a;
    // for each cell, apply the rules of Life
    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++) {
            a = adjacent_to(board, size, i, j);
            if (a == 2) newboard[i][j] = board[i][j];
            if (a == 3) newboard[i][j] = 1;
            if (a < 2) newboard[i][j] = 0;
            if (a > 3) newboard[i][j] = 0;
        }
} */

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

/* print the life board */
void print(cell_t **board, int size) {
    int i, j;
    /* for each row */
    for (j = 0; j < size; j++) {
        /* print each column position... */
        for (i = 0; i < size; i++)
            printf("%c", board[i][j] ? 'x' : ' ');
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
            //c=fgetc(f);
            //putchar(c);
            board[j * size + i] = s[i] == 'x';
        }
        //fscanf(f,"\n");
    }
}

/* read a file into the life board */
void read_file(FILE *f, cell_t **board, int size) {
    int i, j;
    char *s = (char *) malloc(size + 10);

    for (j = 0; j < size; j++) {
        /* get a string */
        fgets(s, size + 10, f);
        /* copy the string to the life board */
        for (i = 0; i < size; i++) {
            //c=fgetc(f);
            //putchar(c);
            board[i][j] = s[i] == 'x';
        }
        //fscanf(f,"\n");
    }
}

int main() {
    int size, flat_size, steps;
    FILE *f;
    cell_t *h_prev;
    int i;

    f = stdin;

    // Read the file and write its content in the host array
    fscanf(f, "%d %d", &size, &steps);
    flat_size = size * size;

    h_prev = allocate_board_flat(flat_size);
    read_file_flat(f, h_prev, size);
    fclose(f);

    dim3 dimGrid(ceil((float) size / 32), ceil((float) size / 32), 1);
    dim3 dimBlock(32, 32, 1);
    cell_t *d_prev, *d_next, *d_tmp;

    /* cudaMallocPitch((void**) &d_prev, &pitch, size * sizeof(cell_t), size);
    cudaMallocPitch((void**) &d_next, &pitch, size * sizeof(cell_t), size);

    cudaMemcpy2D(d_prev, pitch, h_prev, size * sizeof(cell_t), size * sizeof(cell_t), size, cudaMemcpyHostToDevice); */

    gpuErrchk(cudaMalloc((void **) &d_prev, flat_size * sizeof(cell_t)));
    gpuErrchk(cudaMemcpy(d_prev, h_prev, flat_size * sizeof(cell_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void **) &d_tmp, flat_size * sizeof(cell_t)));

    for (i = 0; i < steps; i++) {
        playKernel<<<dimGrid, dimBlock>>>(d_prev, d_next, size);

        cudaMemcpy(d_tmp, d_next, flat_size * sizeof(cell_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_next, d_prev, flat_size * sizeof(cell_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_prev, d_tmp, flat_size * sizeof(cell_t), cudaMemcpyDeviceToDevice);

        // printf("ok");

        /* d_tmp = d_next;
        d_next = d_prev;
        d_prev = d_tmp; */
    }

    // printf("\n");

/* cudaMemcpy2D(d_prev, pitch, h_prev, size * sizeof(cell_t), size * sizeof(cell_t), size, cudaMemcpyDeviceToHost); */

    cudaMemcpy(h_prev, d_prev, flat_size * sizeof(cell_t), cudaMemcpyDeviceToHost);

    // print_flat(h_prev, size);

    free(h_prev);
}
