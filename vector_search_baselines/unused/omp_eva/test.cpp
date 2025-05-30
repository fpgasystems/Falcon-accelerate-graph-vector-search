#include<stdlib.h>
#include<omp.h>
#include<iostream>
#include<vector>

using namespace std;
#define MULTITHREAD 4
#define LOOP 100

int main() {
    int counter = 0;
    vector<vector<int>> V(4);

    #pragma omp parallel num_threads(MULTITHREAD)
    #pragma omp single
    while (counter < 5) {
        for (int j = 0; j < 8; j++) {
            #pragma omp task shared(V)
            {
                int threadid = omp_get_thread_num();
                V[threadid].push_back(j);
            }
        }
        #pragma omp taskwait
        counter++;
    }

    cout << V[0].size() << endl;
    cout << V[1].size() << endl;
    cout << V[2].size() << endl;
    cout << V[3].size() << endl;
}