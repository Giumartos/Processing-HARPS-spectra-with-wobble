/*
Para compilar e fazer a shared library
gcc -Wall -pedantic -O3 -shared -mcmodel=large -o where_array.so where_array.c
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<stdbool.h>

#define LEN 1000000
#define MAX 1000
#define nrep 10

void where(long len_a, float epsilon, char file_array[MAX]){
    /*
    Dada array a[len_a], vai retornar matriz com índices de onde há elementos repetidos. Não inclui a primeira ocorrência.
    
    - long len_a: tamanho da array a ser analisada.
    - float epsilon: intervalo de confiança para procurar repetição (nem sempre valores sao exatamente iguais, diferença na casa decimal).
    - char file_array[MAX]: string com o nome do arquivo que contém a array para procurar as repetições.
    - double a[len]: array com repetições. 
    - long int reps_idx: matriz para guardar os índices das repetições. A linha [i] corresponde ao elemento [i] da array a[len_a], e os
                valores que estão nessas linhas sao os índices onde há repetição desse elemento i. 
    */
    
    long cont = 0; // contador para preencher matriz de indices
    static long int reps_idx[LEN][nrep]; 

    // inicializando a matriz de repeticoes com zeros
    for (long i = 0; i < len_a; i++){ 
        for (int j = 0; j < nrep; j++){ 
            reps_idx[i][j] = 0;
        }
    }
    
    printf("Size of the array: %ld\n\n", len_a);

    static double a[LEN]; // matriz que será lida do arquivo a seguir

    //obtendo array do arquivo
    FILE *file; 
    long i = 0; 

    file = fopen(file_array, "r");  

    // lendo valores do arquivo e colocando na matriz "a"
    while(fscanf(file, "%lf", &a[i])!=EOF){  
    i++;
    }  
    fclose(file); 

    // procurando as repeticoes
    for (long i = 0; i < len_a; i++){  
        // printando o progresso
        if (i % 100 == 0){
            printf("Progress %ld out of %ld.\r", i, len_a);
        }            
        for (long j = 0; j < len_a; j++){        
            if (a[j] <= a[i] + epsilon && a[j] >= a[i] - epsilon && i < j){ // if a[i]-e <= a[j] <= a[i]+e
                reps_idx[i][cont] = j;
                cont++;
            } 
        }
        cont = 0; // quando muda de linha (elemento), reinicia o contador
    }
    printf("\n\nAll superpositions found\n\nSaving to file\n");

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
    // to test the function
    /*
    long len_a = 243236; 
    float epsilon = 0.001;
    where(len_a, epsilon);
    */
    return 0;
}
