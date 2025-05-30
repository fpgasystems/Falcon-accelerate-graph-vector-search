#pragma once

#include <stdint.h>

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

class BloomFilter {

public:
	int num_buckets_;
	int num_hashes_;
	int seed_;
	bool* byte_array_;

  BloomFilter(int num_buckets, int num_hashes, int seed) : 
  	num_buckets_(num_buckets), num_hashes_(num_hashes), seed_(seed){
	byte_array_ = (bool*) malloc(num_buckets);
	memset(byte_array_, 0, num_buckets);
  }

  ~BloomFilter() {
	free(byte_array_);
  }

  void insert(int key) {
	for (int i = 0; i < num_hashes_; i++) {
	  uint32_t hash = MurmurHash2_KeyLen4((char*) &key, i + seed_);
	  uint32_t bucket = hash % num_buckets_;
	  byte_array_[bucket] = true;
	}
  }

  bool contains(int key) {
	for (int i = 0; i < num_hashes_; i++) {
	  uint32_t hash = MurmurHash2_KeyLen4((char*) &key, i + seed_);
	  uint32_t bucket = hash % num_buckets_;
	  if (!byte_array_[bucket]) {
		return false;
	  }
	}
	return true;
  }

  bool check_and_insert(int key) {
	bool exists = contains(key);
	if (!exists) {
	  insert(key);
	}
	return exists;
  }
};
