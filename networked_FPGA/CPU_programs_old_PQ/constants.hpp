#pragma once

#define F2C_PKG_SIZE 1024 // 1024
#define C2F_PKG_SIZE 4096 // 1024
#define G2C_PKG_SIZE 4096 
#define C2G_PKG_SIZE 4096 

#define FPGA_TOPK 100

namespace bit_byte_const {
const size_t bit_int = 32;
const size_t bit_float = 32;
const size_t bit_long_int = 64;
const size_t bit_AXI = 512;

const size_t byte_int = 4;
const size_t byte_float = 4;
const size_t byte_long_int = 8;
const size_t byte_AXI = 64;
} // namespace bit_byte_const

namespace num_packages {
// sizes in number of 512-bit AXI data packet
const size_t AXI_size_C2F_header = 1;
const size_t AXI_size_F2C_header = 1;

inline size_t AXI_size_C2F_query_vector(const size_t D) {
  return D * bit_byte_const::bit_float % bit_byte_const::bit_AXI == 0 ? D * bit_byte_const::bit_float / bit_byte_const::bit_AXI
                                                                      : D * bit_byte_const::bit_float / bit_byte_const::bit_AXI + 1;
}

inline size_t AXI_size_C2F_cell_IDs(const int nprobe) {
  return nprobe * bit_byte_const::bit_int % bit_byte_const::bit_AXI == 0 ? nprobe * bit_byte_const::bit_int / bit_byte_const::bit_AXI
                                                                         : nprobe * bit_byte_const::bit_int / bit_byte_const::bit_AXI + 1;
}

inline size_t AXI_size_C2F_center_vector(const size_t D) {
  return D * bit_byte_const::bit_float % bit_byte_const::bit_AXI == 0 ? D * bit_byte_const::bit_float / bit_byte_const::bit_AXI
                                                                      : D * bit_byte_const::bit_float / bit_byte_const::bit_AXI + 1;
}

inline size_t AXI_size_F2C_vec_ID(const size_t TOPK) {
  return TOPK * bit_byte_const::bit_long_int % bit_byte_const::bit_AXI == 0 ? TOPK * bit_byte_const::bit_long_int / bit_byte_const::bit_AXI
                                                                            : TOPK * bit_byte_const::bit_long_int / bit_byte_const::bit_AXI + 1;
}

inline size_t AXI_size_F2C_dist(const size_t TOPK) {
  return TOPK * bit_byte_const::bit_float % bit_byte_const::bit_AXI == 0 ? TOPK * bit_byte_const::bit_float / bit_byte_const::bit_AXI
                                                                         : TOPK * bit_byte_const::bit_float / bit_byte_const::bit_AXI + 1;
}

} // namespace num_packages