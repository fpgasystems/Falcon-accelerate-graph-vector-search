class MySemaphore {
public:
  int count;
  MySemaphore(int value) : count(value) {}
  MySemaphore() : count(0) {}

  void produce() {
	count++;
	return;
  }

  void consume() {
	while (count == 0) {
	  // wait
	}
	count--;
	return;
  }
};