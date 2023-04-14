/*
Para compilar e fazer a shared library
gcc -Wall -pedantic -O3 -mcmodel=large -o where_array where_array.c
cc -fPIC -shared -o where_array.so where_array.c
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <stdbool.h>

#define LEN 1000000

void where(long len_a, float epsilon){

    /*
    Dada array a[len_a], vai retornar array com indices de onde ha elementos repetidos. Não inclui a primeira ocorrencia.
    
    long len_a: tamanho da array a ser analisada

    float epsilon: intervalo de confiança para procurar repeticao (nem sempre valores sao exatamente iguais, precisao do computador)

    double a[len]: array com repeticoes. 

    int reps_idx: matriz para guardar os indices das repeticoes. A linha [i] corresponde ao elemento [i] da array a, e os
                valores que estao nessas linhas sao os indices das repeticoes
*/

    long cont = 0; // contador para preencher matriz de indices
    int nrep = 100;
    long reps_idx[len_a][nrep]; 

    // inicializando a matriz de repeticoes com zeros
    for (long i = 0; i < len_a; i++){ 
        for (int j = 0; j < nrep; j++){ 
            reps_idx[i][j] = 0;
        }
    }

    static double a[LEN];

    //obtendo array do arquivo

    FILE *file; 
    int i = 0; 

    file = fopen("arquivo com valores de comprimento de onda repetidos.csv", "r");  

    while(fscanf(file, "%lf", &a[i])!=EOF){  
    i++;
    }  
    fclose(file); 


    // procurando as repeticoes
    for (long i = 0; i < len_a; i++){  
        // printando o progresso
        if (i % 10 == 0){
            printf("Progress %ld out of %ld.\r", i, len_a);
        }            
        for (long j = 0; j < len_a; j++){        
            if (a[j] <= a[i] + epsilon && a[j] >= a[i] - epsilon && i < j){ 
                reps_idx[i][cont] = j;
                cont++;
            } 
        }
    }

    // salvando a array com as repeticoes em um arquivo
    FILE *arquivo_saida; 

    arquivo_saida = fopen("reps_matrix.txt", "w");
    
    for (long i = 0; i < len_a; i++){ 
        for (long j = 0; j < nrep; j++){ 
            fprintf(arquivo_saida, "%ld ", reps_idx[i][j]);
        }
        fprintf(arquivo_saida, "\n");
    }

    fclose(arquivo_saida); 

}

int main(){

    return 0;
}
