#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define MIN(a, b)           ((a)<(b)?(a):(b))
#define BLOCK_LOW(id, p, n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id, p, n) \
                     (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(j, p, n) (((p)*((j)+1)-1)/(n))
#define PTR_SIZE           (sizeof(void*))
#define CEILING(i, j)       (((i)+(j)-1)/(j))

#define MAX_ITERATIONS 1

int main(int argc, char** argv){
    int processes, rank;

    FILE* matrix_file;
    int n,m,size,low_value;
    int **matrix=NULL;
    int *matrix_storage=NULL;
    int *vector=NULL;
    int *temp_vector=NULL;
    int *output_vector=NULL;

    MPI_Request request;
    MPI_Status status;
    float elapsed;

#ifdef _DEBUG
    int debug =1;
#endif

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&processes);

    // Reading matrix size
    if (rank == (processes - 1)) {
        matrix_file = fopen("./matrix", "r");
        fscanf(matrix_file, "%d", &m);
        if (processes > m) {
            fprintf(stderr, "The number of processors exceed matrix dimensions\n");
            MPI_Abort(MPI_COMM_WORLD, -2);
        }
        fscanf(matrix_file, "%d", &n);
    }
    MPI_Bcast(&m,1,MPI_INT,processes-1,MPI_COMM_WORLD);
    MPI_Bcast(&n,1,MPI_INT,processes-1,MPI_COMM_WORLD);

    //Preparing array
    size = BLOCK_SIZE(rank, processes, m);
    low_value = BLOCK_LOW(rank, processes, m);
    matrix = (int **) malloc(size * sizeof(int *));
    if(matrix == NULL){
        fprintf(stderr, "Failed allocating matrix\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    matrix_storage = (int *) malloc(size*n*sizeof(int));
    if(matrix_storage == NULL){
        fprintf(stderr, "Failed allocating matrix_storage\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    for(int i=0; i<size; i++){
        matrix[i] = &matrix_storage[i*n];
    }

    vector = (int *) malloc(n* sizeof(int));
    temp_vector = (int *) calloc(size,sizeof(int));
    if(!rank) output_vector = (int *) malloc(m*sizeof(int));
    //Loading array
    if(rank == processes-1){
        for(int i = 0; i < processes-1; i++) {
            int local_size = BLOCK_SIZE(i, processes, m);
            for (int j = 0; j < local_size * n; j++)
                fscanf(matrix_file, "%d", &matrix_storage[j]);
            if (processes > 1) MPI_Irsend(matrix_storage, local_size * n, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
        }
        for (int i = 0; i < size * n; i++)
            fscanf(matrix_file, "%d", &matrix_storage[i]);
        for (int i = 0; i < n; i++){
            fscanf(matrix_file, "%d", &vector[i]);
        }
    } else {
        MPI_Recv(matrix_storage, size * n, MPI_INT, processes - 1, 0, MPI_COMM_WORLD, &status);
    }
    MPI_Bcast(vector,n,MPI_INT,processes-1,MPI_COMM_WORLD);

#ifdef _DEBUG
    if (!rank) {
        fprintf(stdout, "INPUT MATRIX:\n");
        fflush(stdout);
    } else {
        MPI_Recv(&debug, 1, MPI_INT, (rank - 1) % processes, 1, MPI_COMM_WORLD, &status);
    }
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < n; k++) {
            fprintf(stdout, "%d ", matrix[i][k]);
            fflush(stdout);
        }
        fprintf(stdout, "\n");
        fflush(stdout);
    }
    if (processes > 1) MPI_Send(&debug, 1, MPI_INT, (rank + 1) % processes, 1, MPI_COMM_WORLD);
    if(!rank){
        fprintf(stdout,"VECTOR:\n");
        for(int i = 0; i <n; i++)
            fprintf(stdout,"%d ",vector[i]);
        fprintf(stdout,"\n");
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed=-MPI_Wtime();
    for (int iterations = 0; iterations < MAX_ITERATIONS; iterations++) {
        for(int i =0;i<size;i++){
            for(int j=0; j<n; j++){
                temp_vector[i]+= matrix[i][j] * vector[j];
            }
        }
        MPI_Gather(temp_vector,size,MPI_INT,output_vector,m,MPI_INT,0,MPI_COMM_WORLD);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    elapsed+=MPI_Wtime();
    elapsed/=MAX_ITERATIONS;
    MPI_Finalize();

#ifdef _DEBUG
    if(!rank){
        fprintf(stdout,"VECTOR:\n");
        for(int i = 0; i <m; i++)
            fprintf(stdout,"%d ",output_vector[i]);
        fprintf(stdout,"\n");
    }
#endif

    free(matrix),matrix=NULL;
    free(matrix_storage),matrix_storage=NULL;
    free(vector),vector=NULL;
    free(temp_vector), temp_vector=NULL;
    if(!rank)free(output_vector), output_vector=NULL;
    if(rank==processes-1) fclose(matrix_file);

    if(!rank) fprintf(stdout,"Processes %d Iterations: %d Time: %f",processes,MAX_ITERATIONS,elapsed);

    return 0;
}