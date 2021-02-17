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

#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef unsigned char bool_t;
typedef unsigned char cell_t;

cell_t** allocate_board(int size) {
	cell_t** board = (cell_t**)malloc(sizeof(cell_t*) * size);
	int i;
	for (i = 0; i < size; i++)
		board[i] = (cell_t*)malloc(sizeof(cell_t) * size);
	return board;
}

void free_board(cell_t** board, int size) {
	int i;
	for (i = 0; i < size; i++)
		free(board[i]);
	free(board);
}


/* return the number of on cells adjacent to the i,j cell */
int adjacent_to(cell_t** board, int size, int i, int j) {
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
}

void play(cell_t** board, cell_t** newboard, int size) {
	int i, j, a;
	/* for each cell, apply the rules of Life */
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++) {
			a = adjacent_to(board, size, i, j);
			if (a == 2) newboard[i][j] = board[i][j];
			if (a == 3) newboard[i][j] = 1;
			if (a < 2) newboard[i][j] = 0;
			if (a > 3) newboard[i][j] = 0;
		}
}

/* print the life board */
void print(cell_t** board, int size) {
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
void read_file(FILE* f, cell_t** board, int size) {
	int i, j;
	char* s = (char*)malloc(size + 10);

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

void write_file(FILE* f, cell_t** board, int size) {
	int i, j;
	char* s = (char*)malloc(size + 10);

	for (j = 0; j < size; j++) {
		/* print each column position... */
		for (i = 0; i < size; i++)
			fprintf(f, "%c", board[i][j] ? 'x' : ' ');
		/* followed by a carriage return */
		fprintf(f, "\n");
	}
}

void usage() {
	printf("Usage: ./life input_file [output_file]\n");
	printf("input_file: path to the input file\n");
	printf("output_file: path to the output file\n");
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		usage();
		return EXIT_FAILURE;
	}

	int size, steps, i;
	FILE* f_in, * f_out = NULL;
	bool_t writeOutput = 0;
	cell_t** prev, ** tmp, ** next;
	clock_t t;

	// Open files
	printf("Provided input file: %s\n", argv[1]);
	f_in = fopen(argv[1], "r");

	if (argc >= 3) {
		printf("Provided output file: %s\n", argv[2]);
		f_out = fopen(argv[2], "w+");
		writeOutput = 1;
	}

	fscanf(f_in, "%d %d", &size, &steps);
	prev = allocate_board(size);
	read_file(f_in, prev, size);
	fclose(f_in);
	next = allocate_board(size);

#ifdef DEBUG
	printf("Initial \n");
	print(prev, size);
	printf("----------\n");
#endif

	t = clock();

	for (i = 0; i < steps; i++) {
		printf("Step: %d\n", i);
		play(prev, next, size);
#ifdef DEBUG
		printf("%d ----------\n", i);
		print(next, size);
#endif
		tmp = next;
		next = prev;
		prev = tmp;
	}

	t = clock() - t;
	printf("Game of life (%d steps) done in %lf seconds\n", steps, ((double) t) / CLOCKS_PER_SEC);

	if (writeOutput) {
		printf("Writing output file...\n");
		write_file(f_out, prev, size);
		fclose(f_out);
	}

	free_board(prev, size);
	free_board(next, size);
}
