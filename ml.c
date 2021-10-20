/* 
 * 
 * This code calculates the house price of a house by learing from
 * training data. It uses pseudo inverse of a given matrix to find the 
 * weight of different features.
 * 
 * Predicted Price : Y = W0 + W1*x1 + W2*X2 + W3*X3 + W4*X4
 * Weight Matrix : W = pseudoInv(X)*Y
 * pseudoInv(X) = inverse(transpose(X)*X) * transpose(X)  
 * 
 * weight(w) = pseudoInv(X) * Y
 * where X = Input data matrix
 * Y = Target vector
 * 
 */
 
#include<stdio.h>
#include<stdlib.h>

// all methods declarations
double** multiplyMatrix(double** matA, double** matB, int r1, int c1, int r2, int c2);
double** transposeMatrix(double** mat, int row, int col);
double** inverseMatrix(double **matA, int dimension);
void freeMat(double** mat, int row);
void printMat(double** mat, int row, int col);
//double** matrixAllo(int row, int column); 

// main method starts here
int main(int argc, char** argv) {

    FILE *f = fopen(argv[1], "r");
    if(f == NULL)
    {
    	printf("error\n");
	return 0;
    }
    int col;
    int row;
    fscanf(f, "%d\n%d\n", &col, &row);
    col = col + 1;
    double** matX = (double**)malloc(sizeof(double*) * row);
    for(int i = 0; i < row; i++)
    {
    	matX[i] = (double*)malloc(sizeof(double) * col); 
    }
    for(int i = 0; i < row; i++)
    {
    	for(int j = 0; j < col; j++)
	{
		matX[i][j] = 0;
	}
    }
    double** matY = (double**)malloc(sizeof(double*) * row);
    for(int i = 0; i < row; i++)
    {
    	matY[i] = (double*)malloc(sizeof(double)); 
    } 
    for(int i = 0; i < row; i++)
    {	
	matY[i][0] = 0; 
    }
    double p; 
    char stop; 
    for(int i = 0; i < row; i++)
    {
    	for(int j = 0; j < col + 1; j++)
	{
		if(j == 0)
		{
			matX[i][j] = 1; 
		}
		else if(j == col)
		{	
			fscanf(f, "%lf", &p);
			matY[i][0] = p;
		}
		else
		{
			fscanf(f, "%lf", &p);
			fscanf(f, "%c", &stop); 
			matX[i][j] = p; 
		}
	}
	fscanf(f, "\n");
    } 
    fclose(f); 
    FILE *f2 = fopen(argv[2], "r");
    if(f2 == NULL)
    {
    	printf("error\n");
	return 0; 
    }
    int test_data; 
    fscanf(f2, "%d\n", &test_data); 
    double** matXprime = (double**)malloc(sizeof(double*) * test_data); 
    for(int i = 0; i < test_data; i++)
    {
    	matXprime[i] = (double*)malloc(sizeof(double) * col); 
    }
    for(int i = 0; i < test_data; i++)
    {
    	for(int j = 0; j < col; j++)
	{
		matXprime[i][j] = 0; 
	}
    }
    for(int i = 0; i < test_data; i++)
    {
    	for(int j = 0; j < col; j++)
	{
		if(j == 0)
		{
			matXprime[i][j] = 1;
		}
		else 
		{	
			 
			fscanf(f2, "%lf", &p);
			fscanf(f2, "%c", &stop);
			matXprime[i][j] = p; 
		}
	}
	fscanf(f2, "\n"); 
    }
    fclose(f2);   
    /*
    double** transposeX = (double**)malloc(sizeof(double*) * col);
    for(int i = 0; i < col; i++)
    {
    	transposeX[i] = (double*)malloc(sizeof(double) * row); 
    }
    double** multiplyX = (double**)malloc(sizeof(double*) * col);
    for(int i = 0; i < col; i++)
    {
    	multiplyX[i] = (double*)malloc(sizeof(double) * row);
    }
    double** inverseMat = (double**)malloc(sizeof(double*) * col);
    for(int i = 0; i < col; i++)
    {
    	inverseMat[i] = (double*)malloc(sizeof(double) * col); 
    }
    double** inverseMultiply = (double**)malloc(sizeof(double*) * col);
    for(int i = 0; i < col; i++)
    {
    	inverseMultiply[i] = (double*)malloc(sizeof(double) * row); 
    }
    double** W = (double**)malloc(sizeof(double*) * col);
    for(int i = 0; i < col; i++)
    {
    	W[i] = (double*)malloc(sizeof(double) * 1); 
    }
    double** matYprime = (double**)malloc(sizeof(double*) *  row);
    for(int i = 0; i < row; i++)
    {
    	matYprime[i] = (double*)malloc(sizeof(double) * 1); 
    }
    */
    double** transposeX = transposeMatrix(matX, row, col);
    //printMat(transposeX, col, row);
    //printf("X^(T)\n");  
    double** multiplyX = multiplyMatrix(transposeX, matX, col, row, row, col);
    //printMat(multiplyX, col, col);
    //printf("X^(T)*X\n"); 
    double** inverseMat = inverseMatrix(multiplyX, col);
    //printMat(inverseMat, col, col);
    //printf("(X^(T)*X)^(-1)\n"); 
    double** inverseMultiply = multiplyMatrix(inverseMat, transposeX, col, col, col, row);
    //printMat(inverseMultiply, col, row); 
    //printf("(X^(T)*X)^(-1)*X^(T)\n"); 
    double** W = multiplyMatrix(inverseMultiply, matY, col, row, row, 1);
    //printMat(W, col, 1); 
    //printf("W\n");  
    double** matYprime = multiplyMatrix(matXprime, W, test_data, col, col, 1);
    printMat(matYprime, test_data, 1);
    //printf("Y\n"); 
    freeMat(matX, row);
    freeMat(matY, row);
    freeMat(transposeX, col);
    freeMat(multiplyX, col); 
    freeMat(inverseMat, col);
    freeMat(inverseMultiply, col);
    freeMat(W, col);
    freeMat(matXprime, test_data);
    freeMat(matYprime, test_data);  
    return 0;
}

/*
double** matrixAllo(int row, int column)
{
	double** mat = (double**)malloc(sizeof(double*) * row);
	for(int i = 0; i < row; i++)
	{
		mat[i] = (double*)malloc(sizeof(double) * column);
	}
	return mat; 
}
*/

void printMat(double** mat, int row, int col)
{
	for(int i = 0; i < row; i++)
	{	
		for(int j = 0; j < col; j++)
		{
			printf("%0.0lf\t", mat[i][j]);
		}
		printf("\n"); 
	}
}

void freeMat(double** mat, int row)
{
	for(int i = 0; i < row; i++) 
	{
		free(mat[i]); 
	}
	free(mat); 
}

double** multiplyMatrix(double **matA, double **matB, int r1, int c1, int r2, int c2)
{   
    //int product = 0; 
    if(c1 != r2)
    {
    	printf("error\n"); 
	return 0; 
    }
    double** result = malloc(r1*sizeof(double*));
    for(int i = 0; i < r1; i++)
    {
    	result[i] = malloc(c2*sizeof(double)); 
    }
    for(int i = 0; i < r1; i++)
    {
    	for(int j = 0; j < c2; j++)
	{
		result[i][j] = 0;
	}
    }
    for(int i = 0; i < r1; i++)
    {	
	 for(int j = 0; j < c2; j++)
	 {
		for(int k = 0; k < r2; k++)
		{
			result[i][j] += matA[i][k] * matB[k][j];
			//printf("%d\t", product); 
		}
		//product = result[i][j];
		//product = 0; 
	 }
    }
    return result;
}

double** transposeMatrix(double** mat, int row, int col)
{
  
    double** matTran = malloc(col*sizeof(double*));
    for(int i = 0; i < col; i++)
    {
    	matTran[i] = malloc(row*sizeof(double)); 
    } 
    for(int i = 0; i < col; i++)
    {
    	for(int j = 0; j < row; j++)
	{
		matTran[i][j] = 0; 
	}
    }
    for(int i = 0; i < col; i++)
    {
    	for(int j = 0; j < row; j++)
	{
		matTran[i][j] = mat[j][i];
	}
    }
    return matTran;        
}


double** inverseMatrix(double** matA, int dimension)
{ 
    
    double** matI = malloc(dimension*sizeof(double*)); 
    for(int i = 0; i < dimension; i++)
    {
    	matI[i] = malloc(dimension*sizeof(double)); 
    }
    for(int i = 0; i < dimension; i++)
    {
    	for(int j = 0; j < dimension; j++)
	{
		matI[i][j] = 0; 	
	}
    }
    for(int i = 0; i < dimension; i++)
    {
    	for(int j = 0; j < dimension; j++)
	{
		if(i == j)
		{
			matI[i][j] = 1.0;
		}
		else 
		{
			matI[i][j] = 0; 
		}	
	}
    }
    double f = 0.0;   
    for(int i = 0; i < dimension; i++)
    { 	
	f = matA[i][i];
	for(int j = 0; j < dimension; j++)
	{
		matA[i][j] = matA[i][j] / f;
		matI[i][j] = matI[i][j] / f; 
	}
	for(int k = i + 1; k < dimension; k++)
	{
		f = matA[k][i]; 
		for(int l = 0; l < dimension; l++)
		{
			matA[k][l] = matA[k][l] - (matA[i][l] * f);
			matI[k][l] = matI[k][l] - (matI[i][l] * f);  
		}
	}
    }
    for(int i = dimension - 1; i >= 0; i--)
    {
    	for(int k = i - 1; k >= 0; k--) 
	{
		f = matA[k][i]; 
		for(int m = 0; m < dimension; m++)
		{
			matA[k][m] = matA[k][m] - (matA[i][m] * f);
			matI[k][m] = matI[k][m] - (matI[i][m] * f); 
		}
	}
    } 
    return matI;
}


