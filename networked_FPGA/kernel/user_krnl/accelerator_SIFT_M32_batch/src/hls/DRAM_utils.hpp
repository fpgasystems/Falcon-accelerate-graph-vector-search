#pragma once

#include "constants.hpp"
#include "types.hpp"

void get_cell_addr_and_size(
    // in init
    int nlist,
    hls::stream<int> &s_nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_num_vecs,
    // in runtime
	hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_cell_ID,
    // out
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_last_valid_PE_ID,
    hls::stream<int> &s_start_addr_every_cell,
    hls::stream<int> &s_control_iter_num_per_query);

void load_PQ_codes(
    // in runtime
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_cell_ID,
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_last_valid_PE_ID,
    hls::stream<int> &s_start_addr_every_cell,
    const ap_uint<512>* PQ_codes_DRAM_0,
    const ap_uint<512>* PQ_codes_DRAM_1,
    const ap_uint<512>* PQ_codes_DRAM_2,
    const ap_uint<512>* PQ_codes_DRAM_3,
    // out
    hls::stream<PQ_in_t> (&s_PQ_codes)[ADC_PE_NUM]);

void load_meta_data(
    int* meta_data_init,
    hls::stream<int> &s_nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_vec_ID_start_addr,
    hls::stream<int> &s_nlist_num_vecs);



void get_cell_addr_and_size(
    // in init
    int nlist,
    hls::stream<int> &s_nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_num_vecs,
    // in runtime
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_cell_ID,
    // out
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_last_valid_PE_ID,
    hls::stream<int> &s_start_addr_every_cell,
    hls::stream<int> &s_control_iter_num_per_query) {
    // Given the nprobe cell ID per query, output
    //    (1) startin addr in DRAM (entry ID for ap_uint<512>)
    //    (2) number of vectors to scan per cell

    // init a table that maps cell_ID to PQ_code start_address (entry ID for ap_uint<512>)
    int cell_ID_to_addr[NLIST_MAX];
#pragma HLS resource variable=cell_ID_to_addr core=RAM_2P_URAM
    int cell_ID_to_num_vecs[NLIST_MAX]; // number of vectors per cell
#pragma HLS resource variable=cell_ID_to_num_vecs core=RAM_2P_URAM
    for (int i = 0; i < nlist; i++) {
        cell_ID_to_addr[i] = s_nlist_PQ_codes_start_addr.read();
	    cell_ID_to_num_vecs[i] = s_nlist_num_vecs.read();
    }

    while (true) {

        batch_header_t batch_header = s_batch_header.read();
        int batch_size = batch_header.batch_size;
        int nprobe = batch_header.nprobe;
        int terminate = batch_header.terminate;
		int query_num = batch_size; 

        // termination detection
        if (terminate) {
            break;
        }

		for (int query_id = 0; query_id < query_num; query_id++) {

			int iter_num_per_query = 0;
			for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

				int cell_ID = s_cell_ID.read();
				int num_vec = cell_ID_to_num_vecs[cell_ID];

				int scanned_entries_every_cell;
				int last_valid_PE_ID; // from 0 to ADC_PE_NUM
				if (num_vec % ADC_PE_NUM == 0) {
					scanned_entries_every_cell = num_vec / ADC_PE_NUM;
					last_valid_PE_ID = ADC_PE_NUM - 1;
				}
				else {
					scanned_entries_every_cell = num_vec / ADC_PE_NUM + 1;
					last_valid_PE_ID = num_vec % ADC_PE_NUM - 1;
				}
				
				iter_num_per_query += scanned_entries_every_cell;

				s_scanned_entries_every_cell.write(scanned_entries_every_cell);
				s_last_valid_PE_ID.write(last_valid_PE_ID);

				int start_addr = cell_ID_to_addr[cell_ID];
				s_start_addr_every_cell.write(start_addr);
			}
			s_control_iter_num_per_query.write(iter_num_per_query);
		}
	}
}

void load_PQ_codes(
    // in runtime
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_cell_ID,
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_last_valid_PE_ID,
    hls::stream<int> &s_start_addr_every_cell,
    const ap_uint<512>* PQ_codes_DRAM_0,
    const ap_uint<512>* PQ_codes_DRAM_1,
    const ap_uint<512>* PQ_codes_DRAM_2,
    const ap_uint<512>* PQ_codes_DRAM_3,
    // out
    hls::stream<PQ_in_t> (&s_PQ_codes)[ADC_PE_NUM]
) {


    while (true) {

        batch_header_t batch_header = s_batch_header.read();
        int batch_size = batch_header.batch_size;
        int nprobe = batch_header.nprobe;
        int terminate = batch_header.terminate;
		int query_num = batch_size; 

        // termination detection
        if (terminate) {
            break;
        }

		for (int query_id = 0; query_id < query_num; query_id++) {

			for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

				int cell_ID = s_cell_ID.read();
				int compute_iter_per_PE = s_scanned_entries_every_cell.read();
				int last_valid_PE_ID = s_last_valid_PE_ID.read();
				int start_addr = s_start_addr_every_cell.read();

				for (int entry_id = 0; entry_id < compute_iter_per_PE; entry_id++) {
#pragma HLS pipeline II=1 // style=flp
					ap_uint<512> PQ_reg_multi_PE_0 = PQ_codes_DRAM_0[start_addr + entry_id];
					ap_uint<512> PQ_reg_multi_PE_1 = PQ_codes_DRAM_1[start_addr + entry_id];
					ap_uint<512> PQ_reg_multi_PE_2 = PQ_codes_DRAM_2[start_addr + entry_id];
					ap_uint<512> PQ_reg_multi_PE_3 = PQ_codes_DRAM_3[start_addr + entry_id];

					for (int a = 0; a < ADC_PE_PER_CHANNEL; a++) {
#pragma HLS unroll
						int s_ID_0 = a + 0 * ADC_PE_PER_CHANNEL;
						int s_ID_1 = a + 1 * ADC_PE_PER_CHANNEL;
						int s_ID_2 = a + 2 * ADC_PE_PER_CHANNEL;
						int s_ID_3 = a + 3 * ADC_PE_PER_CHANNEL;

						PQ_in_t PQ_reg_0;
						PQ_in_t PQ_reg_1;
						PQ_in_t PQ_reg_2;
						PQ_in_t PQ_reg_3;
						
						PQ_reg_0.valid = ((entry_id == compute_iter_per_PE - 1) && (s_ID_0 > last_valid_PE_ID))? 0 : 1;
						PQ_reg_1.valid = ((entry_id == compute_iter_per_PE - 1) && (s_ID_1 > last_valid_PE_ID))? 0 : 1;
						PQ_reg_2.valid = ((entry_id == compute_iter_per_PE - 1) && (s_ID_2 > last_valid_PE_ID))? 0 : 1;
						PQ_reg_3.valid = ((entry_id == compute_iter_per_PE - 1) && (s_ID_3 > last_valid_PE_ID))? 0 : 1;
						
						PQ_reg_0.cell_ID = cell_ID; 
						PQ_reg_1.cell_ID = cell_ID; 
						PQ_reg_2.cell_ID = cell_ID; 
						PQ_reg_3.cell_ID = cell_ID; 

						PQ_reg_0.offset = entry_id * ADC_PE_PER_CHANNEL + a; // per channel offset
						PQ_reg_1.offset = entry_id * ADC_PE_PER_CHANNEL + a; // per channel offset
						PQ_reg_2.offset = entry_id * ADC_PE_PER_CHANNEL + a; // per channel offset
						PQ_reg_3.offset = entry_id * ADC_PE_PER_CHANNEL + a; // per channel offset

						// bit refer: https://github.com/WenqiJiang/FPGA-ANNS/blob/main/integrated_accelerator/entire-node-systolic-computation-without-FIFO-type-assignment-fine-grained-PE-with-queue-group-inlined/src/HBM_interconnections.hpp
						for (int m = 0; m < M; m++) {
#pragma HLS unroll
							PQ_reg_0.PQ_code[m] = PQ_reg_multi_PE_0.range(
								a * M * 8 + m * 8 + 7, a * M * 8 + m * 8);
							PQ_reg_1.PQ_code[m] = PQ_reg_multi_PE_1.range(
								a * M * 8 + m * 8 + 7, a * M * 8 + m * 8);
							PQ_reg_2.PQ_code[m] = PQ_reg_multi_PE_2.range(
								a * M * 8 + m * 8 + 7, a * M * 8 + m * 8);
							PQ_reg_3.PQ_code[m] = PQ_reg_multi_PE_3.range(
								a * M * 8 + m * 8 + 7, a * M * 8 + m * 8);
						}
						s_PQ_codes[s_ID_0].write(PQ_reg_0);
						s_PQ_codes[s_ID_1].write(PQ_reg_1);
						s_PQ_codes[s_ID_2].write(PQ_reg_2);
						s_PQ_codes[s_ID_3].write(PQ_reg_3);
					}
				}
			}
		}
	}
}

void load_meta_data(
    int nlist,
    int* meta_data_init,
    hls::stream<int> &s_nlist_PQ_codes_start_addr,
    hls::stream<int> &s_nlist_vec_ID_start_addr,
    hls::stream<int> &s_nlist_num_vecs,
    hls::stream<float> &s_product_quantizer_init) {

    // meta_data_init = three of the following 
    // int* nlist_PQ_codes_start_addr,
    // int* nlist_vec_ID_start_addr,
    // int* nlist_num_vecs,
    // float* product_quantizer (M, 256, D/M)

    int offset_nlist_PQ_codes_start_addr = 0;
    int offset_nlist_vec_ID_start_addr = nlist;
    int offset_nlist_num_vecs = 2 * nlist; 
    int offset_product_quantizer = 3 * nlist;

    for (int i = 0; i < nlist; i++) {
#pragma HLS pipeline II=1 // style=flp
        s_nlist_PQ_codes_start_addr.write(meta_data_init[i + offset_nlist_PQ_codes_start_addr]);
        s_nlist_vec_ID_start_addr.write(meta_data_init[i + offset_nlist_vec_ID_start_addr]);
        s_nlist_num_vecs.write(meta_data_init[i + offset_nlist_num_vecs]);
    }

    for (int i = 0; i < D * LUT_ENTRY_NUM; i++) {
#pragma HLS pipeline II=1 // style=flp
        int reg_int = meta_data_init[i + offset_product_quantizer];
        float reg_float = *((float*) (&reg_int));
        s_product_quantizer_init.write(reg_float);
    }

}


