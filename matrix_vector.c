/*
 * Matrix Vector product with Checkerboard Decomposition
 */

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define BLOCK_LOW(id, p, n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id, p, n) \
                     (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)

typedef struct {
    int rows;
    int cols;
} matrix_dims_t;

int main(int argc, char** argv){
    int processes, rank;

    FILE* matrix_file;
    int local_rows,local_cols;
    int **matrix=NULL;
    int *matrix_storage=NULL;
    int *vector=NULL;
    int *temp_vector=NULL;
    int *output_vector=NULL;

    MPI_Request request;
    MPI_Status status;
    float elapsed;
    int *gather_counts=NULL;
    int *gather_displacements=NULL;
    MPI_Comm grid_comm;
    MPI_Comm row_comm;
    MPI_Comm column_comm;
    int cart_dims = 2;
    int cart_size[2]={0,0};
    int wraparaound[2]={0,0};
    int grid_coords[2];
    matrix_dims_t dims;

#ifdef _DEBUG
    int debug =1;
#endif

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&processes);

    MPI_Dims_create(processes,cart_dims,cart_size);
    MPI_Cart_create(MPI_COMM_WORLD,cart_dims,cart_size,wraparaound,1,&grid_comm);

    MPI_Cart_coords (grid_comm, rank, cart_dims, grid_coords);
    MPI_Comm_split (grid_comm, grid_coords[1], grid_coords[0], &column_comm);
    MPI_Comm_split (grid_comm, grid_coords[0], grid_coords[1], &row_comm);

    // Creating MPI Data Type for matrix dims
    MPI_Datatype mpi_matrix_dims_t;
    int lengths[2]={1,1};
    MPI_Aint displacements[2];
    MPI_Aint base_address;
    MPI_Get_address(&dims, &base_address);
    MPI_Get_address(&dims.rows,&displacements[0]);
    MPI_Get_address(&dims.cols,&displacements[1]);
    displacements[0] = MPI_Aint_diff(displacements[0],base_address);
    displacements[1] = MPI_Aint_diff(displacements[1],base_address);
    MPI_Datatype primitive_types[2] = {MPI_INT,MPI_INT};
    MPI_Type_create_struct(2,lengths,displacements,primitive_types,&mpi_matrix_dims_t);
    MPI_Type_commit(&mpi_matrix_dims_t);

    // Reading matrix size
    if (rank == (processes - 1)) {
        matrix_file = fopen("./matrix", "r");
        fscanf(matrix_file, "%d", &dims.rows);
        if (processes > dims.rows) {
            fprintf(stderr, "The number of processors exceed matrix dimensions\n");
            MPI_Abort(MPI_COMM_WORLD, -2);
        }
        fscanf(matrix_file, "%d", &dims.cols);
    }
    MPI_Bcast(&dims,1,mpi_matrix_dims_t,processes-1,MPI_COMM_WORLD);

    //Preparing array
    local_rows = BLOCK_SIZE(rank, processes, dims.rows);
    local_cols = BLOCK_SIZE(rank,processes,dims.cols);
    matrix = (int **) malloc(local_rows * sizeof(int *));
    if(matrix == NULL){
        fprintf(stderr, "Failed allocating matrix\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    matrix_storage = (int *) malloc(local_rows*local_cols*sizeof(int));
    if(matrix_storage == NULL){
        fprintf(stderr, "Failed allocating matrix_storage\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    for(int i=0; i<local_rows; i++){
        matrix[i] = &matrix_storage[i*dims.cols];
    }

    vector = (int *) malloc(dims.cols* sizeof(int));
    temp_vector = (int *) calloc(local_rows,sizeof(int));
    if(rank==processes-1) output_vector = (int *) malloc(dims.rows*sizeof(int));

    //Preparing gather param array

    if(rank == processes-1){
        gather_counts = (int *) malloc(processes*sizeof(int));
        gather_displacements = (int *) malloc(processes*sizeof(int));
        for(int i =0;i<processes;i++){
            gather_counts[i]= BLOCK_SIZE(i,processes,dims.rows);
            gather_displacements[i]= BLOCK_LOW(i,processes,dims.rows);
        }
    }

    //Loading array
    if(rank == processes-1){
        for(int i = 0; i < processes-1; i++) {
            int plocal_rows = BLOCK_SIZE(i, processes, dims.rows);
            int plocal_cols = BLOCK_SIZE(i, processes,dims.cols);

            for (int j = 0; j < plocal_rows * plocal_cols; j++)
                fscanf(matrix_file, "%d", &matrix_storage[j]);
            if (processes > 1) MPI_Irsend(matrix_storage, plocal_rows * plocal_cols, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
        }
        for (int i = 0; i < local_rows * dims.cols; i++)
            fscanf(matrix_file, "%d", &matrix_storage[i]);
        for (int i = 0; i < local_cols; i++){
            fscanf(matrix_file, "%d", &vector[i]);
        }
    } else {
        MPI_Recv(matrix_storage, local_rows * local_cols, MPI_INT, processes - 1, 0, MPI_COMM_WORLD, &status);
    }
    MPI_Bcast(vector,dims.cols,MPI_INT,processes-1,MPI_COMM_WORLD);

#ifdef _DEBUG
    if (!rank) {
        fprintf(stdout, "INPUT MATRIX:\n");
        fflush(stdout);
    } else {
        MPI_Recv(&debug, 1, MPI_INT, (rank - 1) % processes, 1, MPI_COMM_WORLD, &status);
    }
    for (int i = 0; i < local_rows; i++) {
        for (int k = 0; k < dims.cols; k++) {
            fprintf(stdout, "%d ", matrix[i][k]);
            fflush(stdout);
        }
        fprintf(stdout, "\n");
        fflush(stdout);
    }
    if (processes > 1) MPI_Send(&debug, 1, MPI_INT, (rank + 1) % processes, 1, MPI_COMM_WORLD);
    if(!rank){
        fprintf(stdout,"VECTOR:\n");
        for(int i = 0; i <dims.cols; i++)
            fprintf(stdout,"%d ",vector[i]);
        fprintf(stdout,"\n");
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed=-MPI_Wtime();
    for (int iterations = 0; iterations < MAX_ITERATIONS; iterations++) {
        for(int i =0;i<local_rows;i++){
            for(int j=0; j<dims.cols; j++){
                temp_vector[i]+= matrix[i][j] * vector[j];
            }
        }
        MPI_Gatherv(temp_vector,local_rows,MPI_INT,output_vector,gather_counts,gather_displacements,MPI_INT,processes-1,MPI_COMM_WORLD);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    elapsed+=MPI_Wtime();
    elapsed/=MAX_ITERATIONS;
    MPI_Type_free(mpi_matrix_dims_t);
    MPI_Finalize();

#ifdef _DEBUG
    if(rank == processes-1){
        fprintf(stdout,"VECTOR:\n");
        for(int i = 0; i <dims.rows; i++)
            fprintf(stdout,"%d ",output_vector[i]);
        fprintf(stdout,"\n");
    }
#endif


    free(matrix),matrix=NULL;
    free(matrix_storage),matrix_storage=NULL;
    free(vector),vector=NULL;
    free(temp_vector), temp_vector=NULL;
    if(rank==processes-1){
        free(output_vector), output_vector=NULL;
        free(gather_displacements),gather_displacements=NULL;
        free(gather_counts),gather_counts=NULL;
        fclose(matrix_file);
    }
    if(!rank) fprintf(stdout,"Processes %d Iterations: %d Time: %f\n",processes,MAX_ITERATIONS,elapsed);

    return 0;
}