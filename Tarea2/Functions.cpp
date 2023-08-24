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

#define MAX 10000
typedef void (*pf)();
double A[MAX][MAX], x[MAX], y[MAX];
int i=0, j=0;

void TestFunction1(){
    for(i=0; i<MAX; i++){
        for(j=0; j<MAX; j++){
            y[i] += A[i][j]*x[j];
        }
    }
}
void TestFunction2(){
    for(j=0; j<MAX; j++){
        for(i=0; i<MAX; i++){
            y[i] += A[i][j]*x[j];
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
    
    pf fun1[] = {TestFunction1, TestFunction2};
    for(int i=0; i<2; i++) {
        elapse(fun1[i]);
    }
    
    return 0;
}
