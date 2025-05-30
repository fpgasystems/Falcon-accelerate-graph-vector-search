# accelerator_flush_explicit

LUT construction + PQ Scan + K-selection. Here, PQ scan has the option of enabling double buffering (which typically cause P&R error with network). Also the results are fully sorted.

See entire_accelerator_final for the version without network.

## Register flush settings 

default setting: Free-Running/ Flushable Pipeline

There are several places connected to AXI, where only Flushable Pipeline is the option, so I explicitly set the style for them:

```
pipelineConfig.tcl:# config_compile -pipeline_style flp 
src/hls/DRAM_utils.hpp:#pragma HLS pipeline II=1 style=flp
src/hls/DRAM_utils.hpp:#pragma HLS pipeline II=1 style=flp
src/hls/DRAM_utils.hpp:#pragma HLS pipeline II=1 style=flp
src/hls/hierarchical_priority_queue.hpp:#pragma HLS pipeline II=1 style=flp
```