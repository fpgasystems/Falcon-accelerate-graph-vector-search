#include<stdlib.h>
#include<omp.h>
#include<iostream>

using namespace std;
#define MULTITHREAD 24
#define LOOP 100

int main() {
    double itime, ftime, timetaken = 0;
    int i;

    // without workload
    for (i = 0; i < LOOP; i++) {
        itime = omp_get_wtime();
        #pragma omp parallel num_threads(MULTITHREAD)
        {
        }
        ftime = omp_get_wtime();
        // cout << (ftime - itime) * 1e6 << endl;
        timetaken += (ftime - itime) * 1e6;
    }
    cout << "Time taken: " << timetaken / LOOP << endl;
}