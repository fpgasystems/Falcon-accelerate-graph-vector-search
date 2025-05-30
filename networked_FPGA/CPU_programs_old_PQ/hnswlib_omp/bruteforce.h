#pragma once
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>

#include <omp.h>

namespace hnswlib {
    template<typename dist_t>
    class BruteforceSearch : public AlgorithmInterface<dist_t> {
    public:
        BruteforceSearch(SpaceInterface <dist_t> *s) {

        }
        BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location) {
            loadIndex(location, s);
        }

        BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements) {
            maxelements_ = maxElements;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *) malloc(maxElements * size_per_element_);
            if (data_ == nullptr)
                std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
            cur_element_count = 0;
        }

        ~BruteforceSearch() {
            free(data_);
        }

        char *data_;
        size_t maxelements_;
        size_t cur_element_count;
        size_t size_per_element_;

        size_t data_size_;
        DISTFUNC <dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::mutex index_lock;

        std::unordered_map<labeltype,size_t > dict_external_to_internal;

        void addPoint(const void *datapoint, labeltype label) {

            int idx;
            {
                std::unique_lock<std::mutex> lock(index_lock);



                auto search=dict_external_to_internal.find(label);
                if (search != dict_external_to_internal.end()) {
                    idx=search->second;
                }
                else{
                    if (cur_element_count >= maxelements_) {
                        throw std::runtime_error("The number of elements exceeds the specified limit\n");
                    }
                    idx=cur_element_count;
                    dict_external_to_internal[label] = idx;
                    cur_element_count++;
                }
            }
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);




        };

        void removePoint(labeltype cur_external) {
            size_t cur_c=dict_external_to_internal[cur_external];

            dict_external_to_internal.erase(cur_external);

            labeltype label=*((labeltype*)(data_ + size_per_element_ * (cur_element_count-1) + data_size_));
            dict_external_to_internal[label]=cur_c;
            memcpy(data_ + size_per_element_ * cur_c,
                   data_ + size_per_element_ * (cur_element_count-1),
                   data_size_+sizeof(labeltype));
            cur_element_count--;

        }


		// single query search, sequential 
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void *query_data, size_t k) const {
            std::priority_queue<std::pair<dist_t, labeltype >> topResults;
            if (cur_element_count == 0) return topResults;
            for (int i = 0; i < k; i++) {
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype *) (data_ + size_per_element_ * i +
                                                                                    data_size_))));
            }
            dist_t lastdist = topResults.top().first;
            for (int i = k; i < cur_element_count; i++) {
                dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                if (dist <= lastdist) {
                    topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype *) (data_ + size_per_element_ * i +
                                                                                        data_size_))));
                    if (topResults.size() > k)
                        topResults.pop();
                    lastdist = topResults.top().first;
                }

            }
            return topResults;
        };

		// batch search, without parallelism
        std::vector<std::priority_queue<std::pair<dist_t, labeltype >>>
        searchKnnBatch(const void *query_data, size_t k, int batch_size) const {

			std::vector<std::priority_queue<std::pair<dist_t, labeltype >>> topResultsBatch;

			for (int b = 0; b < batch_size; b++) {
				topResultsBatch.push_back(searchKnn((char*) query_data + b * data_size_, k));
			}
			return topResultsBatch;	
		}


		// single query search, with intra-query parallelism
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnnParallel(const void *query_data, size_t k) const {

			int num_threads = omp_get_max_threads();
			// std::cout << num_threads << std::endl;

            std::priority_queue<std::pair<dist_t, labeltype >> topResultsMerged;
			if (cur_element_count == 0) return topResultsMerged;

			int count_per_thread = cur_element_count / num_threads; // last thread takes the leftover
			dist_t lastdistMerged;

			// each thread handle a part of the 
			#pragma omp parallel for schedule(static)
			for (int t = 0; t < num_threads; t++) {

				int count_this_thread = (t == num_threads - 1) ? cur_element_count - (num_threads - 1) * count_per_thread : count_per_thread;
				int init_k = k <= count_this_thread ? k : count_this_thread;
				int start_id = t * count_this_thread;
				int end_id = start_id + count_this_thread;

				// first k 
				std::priority_queue<std::pair<dist_t, labeltype >> topResults;
				for (int i = start_id; i < start_id + init_k; i++) {
					dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
					topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype *) (data_ + size_per_element_ * i +
																						data_size_))));
				}
				dist_t lastdist = topResults.top().first;

				// k ~ rest per thread
				for (int i = start_id + init_k; i < end_id; i++) {
					dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
					if (dist <= lastdist) {
						topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype *) (data_ + size_per_element_ * i +
																							data_size_))));
						if (topResults.size() > k)
							topResults.pop();
						lastdist = topResults.top().first;
					}
				}

				// merge to the main queue
				#pragma omp critical
				{
					while (topResultsMerged.size() < k && topResults.size() > 0) {
						topResultsMerged.push(topResults.top());
						topResults.pop();	
					}
					lastdistMerged = topResultsMerged.top().first;

					while (topResults.size() > 0) {
						if (topResults.top().first < lastdistMerged) {
							topResultsMerged.push(topResults.top());
							if (topResultsMerged.size() > k)
								topResultsMerged.pop();
							lastdistMerged = topResultsMerged.top().first;
						}
						topResults.pop();	
					}
				}
			}
            return topResultsMerged;
        };

        std::vector<std::priority_queue<std::pair<dist_t, labeltype >>>
        searchKnnBatchParallel(const void *query_data, size_t k, int batch_size) const {

			int num_threads = omp_get_max_threads();
			// std::cout << num_threads << std::endl;
			// std::cout << batch_size << std::endl;

			std::vector<std::priority_queue<std::pair<dist_t, labeltype >>> topResultsBatch(batch_size);

			// automatically choose parallel mode: inter-query parallel and intra-query parallel,
			//   given the batch sizes and OMP thread num avaialble
			if (batch_size >= num_threads) {
				// inter-query parallel
				#pragma omp parallel for schedule(guided)
				for (int b = 0; b < batch_size; b++) {
					topResultsBatch.at(b) = searchKnn((char*) query_data + b * data_size_, k);
				}
			} else {
				// intra-query parallel
				for (int b = 0; b < batch_size; b++) {
					topResultsBatch.at(b) = searchKnnParallel((char*) query_data + b * data_size_, k);
				}
			}
			return topResultsBatch;	
		}

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);

            output.write(data_, maxelements_ * size_per_element_);

            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {


            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *) malloc(maxelements_ * size_per_element_);
            if (data_ == nullptr)
                std::runtime_error("Not enough memory: loadIndex failed to allocate data");

            input.read(data_, maxelements_ * size_per_element_);

            input.close();

        }

    };
}
