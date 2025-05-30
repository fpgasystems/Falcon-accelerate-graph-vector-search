# Priority queue performance and correctness test

Run the `insert_sort` function N times and the `pop_top` function for M times. We can then set either N or M as 0 to test the performance of the other function.

## Insert & Sort Performance

Performance & Correctness as expected:

Insertion + sort cycles = input_array_size * 2 + runtime_queue_size

We use runtime_queue_size = [128, 64, 16] and input_array_size = [256, 16] to test.

```
int runtime_queue_size = 128;
int input_array_size = 256;
int iter_insert_sort = 10000;

expected cycles = (256 * 2 + 128) * 10000 = 6,400,000 cycles 
actual cycles = 32.2544/ 1000 * 2e8 = 6,450,880 cycles -> great! 


int runtime_queue_size = 128;
int input_array_size = 16;
int iter_insert_sort = 10000;

expected cycles = (16 * 2 + 128) * 10000 = 1,600,000 cycles 
actual cycles = 8.2522/ 1000 * 2e8 = 1,650,440 cycles -> great! 


int runtime_queue_size = 64;
int input_array_size = 16;
int iter_insert_sort = 10000;

expected cycles = (16 * 2 + 64) * 10000 = 960,000 cycles 
actual cycles = 5.05248 / 1000 * 2e8 = 1,010,496 cycles -> great! 


int runtime_queue_size = 64;
int input_array_size = 256;
int iter_insert_sort = 10000;

expected cycles = (256 * 2 + 64) * 10000 = 5,760,000 cycles 
actual cycles = 29.0532 / 1000 * 2e8 = 5,810,640 cycles -> great!


int runtime_queue_size = 16;
int input_array_size = 256;
int iter_insert_sort = 10000;

expected cycles = (256 * 2 + 16) * 10000 = 5,280,000 cycles 
actual cycles = 26.6541 / 1000 * 2e8 = 5,330,820 cycles -> great!


int runtime_queue_size = 16;
int input_array_size = 16;
int iter_insert_sort = 10000;

expected cycles = (16 * 2 + 16) * 10000 = 480,000 cycles 
actual cycles = 2.65194 / 1000 * 2e8 = 530,388 cycles -> great!
```

### Pop Performance

Pop II = 1 cycle, latency = 2 cycles

From HLS report: Final II = 1, Depth = 2

```
int iter_insert_sort = 0;
int iter_pop = 1000 * 1000;

expected cycles = 1 * 1e6 cycles 
actual cycles = 5.0023 / 1000 * 2e8 = 1,000,460 cycles -> great!
```

Correctness untested yet.