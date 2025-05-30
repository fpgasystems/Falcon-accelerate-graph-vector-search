# https://docs.xilinx.com/r/2021.2-English/ug1399-vitis-hls/Running-Vitis-HLS-from-the-Command-Line
open_project hls_output 
open_solution xcu250-figd2104-2L-e
# open_solution xcu280-fsvh2892-2L-e  
add_files -cflags "-std=c++14" src/vadd.cpp 
add_files -tb src/test.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
set_top vadd 
set_part xcu250-figd2104-2L-e
create_clock -period 140MHz
config_interface -m_axi_addr64
# csim_design
csynth_design
cosim_design
exit
