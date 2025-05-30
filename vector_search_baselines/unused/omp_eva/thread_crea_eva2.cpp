#include<stdlib.h>
#include<omp.h>
#include<iostream>

using namespace std;
#define MULTITHREAD 16
#define LOOP 100

int main() {
    double itime, ftime, timetaken = 0;
    double itime_, ftime_, timetaken_ = 0;
    int i, j;

    // with workload
    for (i = 0; i < LOOP; i++) {
        itime = omp_get_wtime();
        #pragma omp parallel num_threads(MULTITHREAD)
        {   
            #pragma omp single
            {
                itime_ = omp_get_wtime();
                for (j = 0; j < 1000000; j++) {
                    int k = j * j;
                }
                ftime_ = omp_get_wtime();
                // cout << (ftime_ - itime_) * 1e6 << endl;
                timetaken_ += (ftime_ - itime_) * 1e6;
            }

        }
        ftime = omp_get_wtime();
        // cout << (ftime - itime) * 1e6 << endl;
        timetaken += (ftime - itime) * 1e6;
    }
    cout << "Time taken (workload): " << timetaken_ / LOOP << endl;
    cout << "Time taken: " << timetaken / LOOP << endl;
    cout << "Time take (creation): " << (timetaken - timetaken_) / LOOP << endl;
}