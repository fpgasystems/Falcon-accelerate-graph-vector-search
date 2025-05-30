// reference: https://embeddedartistry.com/blog/2017/05/17/creating-a-circular-buffer-in-c-and-c/

class RingBuffer {
/* 
A ring buffer,
  per slot size: per_slot_bytes (bytes)
  number of slots: num_slots

  head_ and tail_ are the indices of the next slot to be written to and read from, respectively.
*/

public:

	// std::mutex mutex_;
	size_t head_ = 0;
	size_t tail_ = 0;
	bool full_ = 0;

  char** buf_;
  const size_t bytes_per_slot_;
  const size_t num_slots_;

	RingBuffer(size_t in_bytes_per_slot, size_t in_num_slots) :
		bytes_per_slot_(in_bytes_per_slot), num_slots_(in_num_slots)
	{
    buf_ = new char*[num_slots_];
    for (size_t i = 0; i < num_slots_; i++) {
      buf_[i] = new char[bytes_per_slot_];
    }
	}


	void reset()
	{
		// std::lock_guard<std::mutex> lock(mutex_);
		head_ = tail_;
		full_ = false;
	}

	bool empty() const
	{
		//if head and tail are equal, we are empty
		return (!full_ && (head_ == tail_));
	}

	bool full() const
	{
		//If tail is ahead the head by 1, we are full
		return full_;
	}

	size_t capacity() const
	{
		return num_slots_;
	}

	size_t size() const
	{
		size_t size = num_slots_;

		if(!full_)
		{
			if(head_ >= tail_)
			{
				size = head_ - tail_;
			}
			else
			{
				size = num_slots_ + head_ - tail_;
			}
		}

		return size;
	}

  void write_slot(char* input_buf)
	{
		// std::lock_guard<std::mutex> lock(mutex_);

    // memcpy
    memcpy(buf_[head_], input_buf, bytes_per_slot_);

		if(full_)
		{
			tail_ = (tail_ + 1) % num_slots_;
		}

		head_ = (head_ + 1) % num_slots_;

		full_ = head_ == tail_;
	}

	void read_slot(char* output_buf)
	{
		// std::lock_guard<std::mutex> lock(mutex_);

		if(empty())
		{
      return;
		} else {
      // memcpy
      memcpy(output_buf, buf_[tail_], bytes_per_slot_);
    }

		//Read data and advance the tail (we now have a free space)
		full_ = false;
		tail_ = (tail_ + 1) % num_slots_;
	}
};