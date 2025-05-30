
### Build

Make hardware: 

```
# U250:
time make all TARGET=hw PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1 > out_hw 2>&1

# U55c: 
time make all TARGET=hw PLATFORM=xilinx_u55c_gen3x16_xdma_3_202210_1 > out_hw 2>&1
```

Make host:

```
make host
// g++ -g -std=c++11 -I/home/wejiang/opt/xilinx/xrt/include -o host src/host.cpp -L/home/wejiang/opt/xilinx/xrt/lib -lxilinxopencl -pthread -lrt
```

### Emulation

Make emu:

```
time make all TARGET=hw_emu PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1 > out_hw_emu 2>&1
```

If enable GUI:

```
[Emulation]
debug_mode=gui 
```

Run emu:

```
make run TARGET=hw_emu PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1  > out_run_hw_emu 2>&1
```

Kill all emulations:

```
ps aux | grep -i hw_emu | grep wejiang | awk '{print $2}' | xargs -i kill -9 {}  
rm *.log # remove emulation logs
```

### Kill

Kill all Xilinx / Vitis process (prevent failure in removing files):

Kill everything:

```
# careful when using this, don't use during build
ps aux | grep wejiang | awk '{print $2}' | xargs -i kill -9 {}  
```

```
ps aux | grep -i xilinx | grep wejiang | awk '{print $2}' | xargs -i kill -9 {}  
```

Reset FPGA state:

```
xbutil reset --device
```

Validate FPGA:

```
xbutil validate --device
```

### Check report

Check resource utilization:

```
vi _x.hw/link/vivado/vpl/prj/prj.runs/impl_1/full_util_placed.rpt

vi _x.hw/vadd.hw/vadd/vadd/solution/syn/report/vadd_csynth.rpt 
```

For networked project:

```
vi build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/link/vivado/vpl/prj/prj.runs/impl_1/full_util_placed.rpt
```