//
//  main.cpp
//  Performance
//
//  Created by Jesus on 22/08/23.
//

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define MAX 1000
typedef void (*pf)();
double A[MAX][MAX], x[MAX][MAX];
int i=0, j=0, k=0;
int ii=0, jj=0, kk=0;

void loadData(){
    A[0][0] = 1;
    A[0][1] = 2;
    A[0][2] = 3;
    A[0][3] = 4;
    A[0][4] = 5;
    A[1][0] = 1;
    A[1][1] = 2;
    A[1][2] = 3;
    A[1][3] = 4;
    A[1][4] = 5;
    A[2][0] = 1;
    A[2][1] = 2;
    A[2][2] = 3;
    A[2][3] = 4;
    A[2][4] = 5;
    A[3][0] = 1;
    A[3][1] = 2;
    A[3][2] = 3;
    A[3][3] = 4;
    A[3][4] = 5;
    A[4][0] = 1;
    A[4][1] = 2;
    A[4][2] = 3;
    A[4][3] = 4;
    A[4][4] = 5;
    
    for(int ij=0;ij<MAX; ij++){
        for(int jk=0;jk<MAX; jk++){
            x[ij][jk] = 0;
        }
    }
}

void print(){
    std::cout << "Print:" << std::endl;
    for(int ij=0;ij<MAX; ij++){
        for(int jk=0;jk<MAX; jk++){
            std::cout << x[ij][jk] << " ";
        }
        std::cout << std::endl;
    }
}

void multiplicacion(){
    for(i=0; i<MAX; i++){
        for(j=0; j<MAX; j++){
            for(k=0; k<MAX; k++){
                x[i][j] += (A[i][k] * A[k][j]);
            }
        }
    }
}
void blockMultiplicacion(){
    int b = 4; // stride = 2^n
    for(ii=0; ii<MAX; ii+=b){
        for(jj=0; jj<MAX; jj+=b){
            for(kk=0; kk<MAX; kk+=b){
                for(i=ii; i<std::min(ii+b-1, MAX); i++){
                    for(j=jj; j<std::min(jj+b-1, MAX); j++){
                        for(k=kk; k<std::min(kk+b-1, MAX); k++){
                            x[i][j] += (A[i][k] * A[k][j]);
                        }
                    }
                }
            }
        }
    }
}

void elapse(pf func){
    auto start = std::chrono::high_resolution_clock::now();
    func( );
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    std::cout << "end for " << duration.count() << " us" << std::endl;
}

int main(int argc, const char * argv[]) {
    
    pf fun1[] = {multiplicacion, blockMultiplicacion};
    for(int i=0; i<2; i++) {
//        loadData();
        elapse(fun1[i]);
//        print();
    }
    
    return 0;
}
