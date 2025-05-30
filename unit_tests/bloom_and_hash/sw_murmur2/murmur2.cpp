// g++ -o murmur2 murmur2.cpp && ./murmur2

#include <iostream>
#include <stdint.h>
#include <chrono>

// https://github.com/abrandoned/murmur2/blob/master/MurmurHash2.c
uint32_t MurmurHash2 ( const void * key, int len, uint32_t seed )
{
  /* 'm' and 'r' are mixing constants generated offline.
     They're not really 'magic', they just happen to work well.  */

  const uint32_t m = 0x5bd1e995;
  const int r = 24;

  /* Initialize the hash to a 'random' value */

  uint32_t h = seed ^ len;

  /* Mix 4 bytes at a time into the hash */

  const unsigned char * data = (const unsigned char *)key;

  while(len >= 4)
  {
    uint32_t k = *(uint32_t*)data;

    k *= m;
    k ^= k >> r;
    k *= m;

    h *= m;
    h ^= k;

    data += 4;
    len -= 4;
  }

  /* Handle the last few bytes of the input array  */

  switch(len)
  {
  case 3: h ^= data[2] << 16;
  case 2: h ^= data[1] << 8;
  case 1: h ^= data[0];
      h *= m;
  };

  /* Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.  */

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
} 


uint32_t MurmurHash2_KeyLen4 ( const void * key, uint32_t seed )
{
  /* 'm' and 'r' are mixing constants generated offline.
     They're not really 'magic', they just happen to work well.  */

  const uint32_t m = 0x5bd1e995;
  /* Initialize the hash to a 'random' value */


  /* Mix 4 bytes at a time into the hash */
	uint32_t k = *(uint32_t*)key;
	k *= m;
	k ^= k >> 24;
	k *= m;

  const int len = 4;
  uint32_t h = seed ^ len;
	h *= m;
	h ^= k;

  /* Handle the last few bytes of the input array  */

  /* Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.  */

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
} 

uint32_t MurmurHash2_KeyLen3 ( const void * key, uint32_t seed )
{
  /* 'm' and 'r' are mixing constants generated offline.
     They're not really 'magic', they just happen to work well.  */

  const uint32_t m = 0x5bd1e995;

  /* Initialize the hash to a 'random' value */

  const int len = 3;
  uint32_t h = seed ^ len;

  /* Mix 4 bytes at a time into the hash */

  const unsigned char * data = (const unsigned char *)key;

  /* Handle the last few bytes of the input array  */
  h ^= data[2] << 16;

  /* Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.  */
  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
} 

void eval_throughput(int num_keys, uint32_t seed){
  // throughput test for MurmurHash2_KeyLen4, record time
  std::cout << std::endl << "Throughput test for MurmurHash2_KeyLen4" << std::endl;
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_keys; i++) {
    int key = i;
    MurmurHash2_KeyLen4((char*) &key, seed);
  }
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "Time for " << num_keys << " keys: " << time_span.count() << " seconds." << std::endl;
  std::cout << "Throughput: " << num_keys / time_span.count() << " keys/second." << std::endl;

}
void check_distribution_balance(uint32_t num_keys, uint32_t seed, uint32_t num_buckets) {
  // check how balanced the distribution is, given 1024 buckets and 1M keys
  std::cout << std::endl << "Check how balanced the distribution is, given 1024 buckets and 1M keys" << std::endl;
  int* buckets = (int*) malloc(sizeof(int) * num_buckets);
  for (int i = 0; i < num_buckets; i++) {
    buckets[i] = 0;
  }
  for (int i = 0; i < num_keys; i++) {
    int key = i;
    int hash = MurmurHash2_KeyLen4((char*) &key, seed);
    int bucket = hash % num_buckets;
    if (bucket >= num_buckets || bucket < 0) {
      std::cout << "bucket = " << bucket << std::endl;
    }
    buckets[bucket]++;
  }
  int max_bucket = 0;
  int min_bucket = num_keys;
  int total_count = 0;
  for (int i = 0; i < num_buckets; i++) {
    if (buckets[i] > max_bucket) {
      max_bucket = buckets[i];
    }
    if (buckets[i] < min_bucket) {
      min_bucket = buckets[i];
    }
    total_count += buckets[i];
  }
  std::cout << "Max bucket size: " << max_bucket << std::endl;
  std::cout << "Min bucket size: " << min_bucket << std::endl;
  std::cout << "Max / Min: " << (double) max_bucket / min_bucket << std::endl;
  std::cout << "Total count: " << total_count << std::endl;
  free(buckets);
}  


int main() {

	int seed = 1;
	// 1000 keys from 0 to 999
	for (int i = 0; i < 10; i++) {
		int key = i;
    std::cout << std::endl << "i = " << i << std::endl;
		std::cout << "MurmurHash2: " << MurmurHash2((char*) &key, 4, seed) << std::endl;
		std::cout << "MurmurHash2_KeyLen4" << MurmurHash2_KeyLen4((char*) &key, seed) << std::endl;
    int key_3_bytes = key << 8;
		std::cout << "MurmurHash2_KeyLen3 (incorrect)" << MurmurHash2_KeyLen3((char*) &key_3_bytes, seed) << std::endl;
	}

  int num_buckets = 1000;
  int num_keys = 1000 * 1000;
  eval_throughput(num_keys, seed);

  check_distribution_balance(num_keys, seed, num_buckets);

	return 0;
}
