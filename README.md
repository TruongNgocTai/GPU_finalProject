# GPU_finalProject
Final project GPU

The basic idea behind a radix sort is that we will consider each element to be sorted digit by digit, from least significant to most significant. For each digit, we will move the elements so that those digits are in increasing order.

Let's take a really simple example. Let's sort four quantities, each of which have 4 binary digits. Let's choose 1, 4, 7, and 14. We'll mix them up and also visualize the binary representation:
```
Element #    1       2       3       4
Value:       7       14      4       1
Binary:      0111    1110    0100    0001
```
First we will consider bit 0:
```
Element #    1       2       3       4
Value:       7       14      4       1
Binary:      0111    1110    0100    0001
bit 0:       1       0       0       1
```
Now the radix sort algorithm says we must move the elements in such a way that (considering only bit 0) all the zeroes are on the left, and all the ones are on the right. Let's do this while preserving the order of the elements with a zero bit and preserving the order of the elements with a one bit. We could do that like this:
```
Element #    2       3       1       4
Value:       14      4       7       1
Binary:      1110    0100    0111    0001
bit 0:       0       0       1       1
```
The first step of our radix sort is complete. The next step is to consider the next (binary) digit:
```
Element #    3       2       1       4
Value:       4       14      7       1
Binary:      0100    1110    0111    0001
bit 1:       0       1       1       0
```
Once again, we must move elements so that the digit in question (bit 1) is arranged in ascending order:
```
Element #    3       4       2       1
Value:       4       1       14      7
Binary:      0100    0001    1110    0111
bit 1:       0       0       1       1
```
Now we must move to the next higher digit:
```
Element #    3       4       2       1
Value:       4       1       14      7
Binary:      0100    0001    1110    0111
bit 2:       1       0       1       1
```
And move them again:
```
Element #    4       3       2       1
Value:       1       4       14      7
Binary:      0001    0100    1110    0111
bit 2:       0       1       1       1
```
Now we move to the last (highest order) digit:
```
Element #    4       3       2       1
Value:       1       4       14      7
Binary:      0001    0100    1110    0111
bit 3:       0       0       1       0
```
And make our final move:
```
Element #    4       3       1       2
Value:       1       4       7       14
Binary:      0001    0100    0111    1110
bit 3:       0       0       0       1
```
And the values are now sorted. This hopefully seems clear, but in the description so far we've glossed over the details of things like "how do we know which elements to move?" and "how do we know where to put them?" So let's repeat our example, but we'll use the specific methods and sequence suggested in the prompt, in order to answer these questions. Starting over with bit 0:
```
Element #    1       2       3       4
Value:       7       14      4       1
Binary:      0111    1110    0100    0001
bit 0:       1       0       0       1
```
First let's build a histogram of the number of zero bits in bit 0 position, and the number of 1 bits in bit 0 position:
```
bit 0:       1       0       0       1

              zero bits       one bits
              ---------       --------
1)histogram:         2              2
```
Now let's do an exclusive prefix-sum on these histogram values:
```
              zero bits       one bits
              ---------       --------
1)histogram:         2              2
2)prefix sum:        0              2
```
An exclusive prefix-sum is just the sum of all preceding values. There are no preceding values in the first position, and in the second position the preceding value is 2 (the number of elements with a 0 bit in bit 0 position). Now, as an independent operation, let's determine the relative offset of each 0 bit amongst all the zero bits, and each one bit amongst all the one bits:
```
bit 0:       1       0       0       1
3)offset:    0       0       1       1
```
This can actually be done programmatically using exclusive prefix-sums again, considering the 0-group and 1-group separately, and treating each position as if it has a value of 1:
```
0 bit 0:             1       1       
3)ex. psum:          0       1    

1 bit 0:     1                        1      
3)ex. psum:  0                        1   
```
Now, step 4 of the given algorithm says:

4) Combine the results of steps 2 & 3 to determine the final output location for each element and move it there

What this means is, for each element, we will select the histogram-bin prefix sum value corresponding to its bit value (0 or 1) and add to that, the offset associated with its position, to determine the location to move that element to:
```
Element #    1       2       3       4
Value:       7       14      4       1
Binary:      0111    1110    0100    0001
bit 0:       1       0       0       1
hist psum:   2       0       0       2
offset:      0       0       1       1
new index:   2       0       1       3
```
Moving each element to its "new index" position, we have:
```
Element #    2       3       1       4
Value:       14      4       7       1
Binary:      0111    1110    0111    0001
```
Which is exactly the result we expect for the completion of our first digit-move, based on the previous walk-through. This has completed step 1, i.e. the first (least-significant) digit; we still have the remaining digits to process, creating a new histogram and new prefix sums at each step.

Notes:

Radix-sort, even in a computer, does not have to be done based strictly on binary digits. It's possible to construct a similar algorithm with digits of different sizes, perhaps consisting of 2,3, or 4 bits.
One of the optimizations we can perform on a radix sort is to only sort based on the number of digits that are actually meaningful. For example, if we are storing quantities in 32-bit values, but we know that the largest quantity present is 1023 (2^10-1), we need not sort on all 32 bits. We can stop, expecting a proper sort, after proceeding through the first 10 bits.
What does any of this have to do with GPUs? In so far as the above description goes, not much. The practical application is to consider using parallel algorithms for things like the histogram, the prefix-sums, and the data movement. This decomposition of radix-sort allows one to locate and use parallel algorithms already developed for these more basic operations, in order to construct a fast parallel sort.
What follows is a worked example. This may help with your understanding of radix sort. I don't think it will help with your assignment, because this example performs a 32-bit radix sort at the warp level, for a single warp, ie. for 32 quantities. But a possible advantage from an understanding point of view is that things like histogramming and prefix sums can be done at the warp level in just a few instructions, taking advantage of various CUDA intrinsics. For your assignment, you won't be able to use these techniques, and you will need to come up with full-featured parallel prefix sums, histograms, etc. that can operate on an arbitrary dataset size.
``` cuda
#include <stdio.h>
#include <stdlib.h>
#define WSIZE 32
#define LOOPS 100000
#define UPPER_BIT 31
#define LOWER_BIT 0

__device__ unsigned int ddata[WSIZE];

// naive warp-level bitwise radix sort

__global__ void mykernel(){
  __shared__ volatile unsigned int sdata[WSIZE*2];
  // load from global into shared variable
  sdata[threadIdx.x] = ddata[threadIdx.x];
  unsigned int bitmask = 1<<LOWER_BIT;
  unsigned int offset  = 0;
  unsigned int thrmask = 0xFFFFFFFFU << threadIdx.x;
  unsigned int mypos;
  //  for each LSB to MSB
  for (int i = LOWER_BIT; i <= UPPER_BIT; i++){
    unsigned int mydata = sdata[((WSIZE-1)-threadIdx.x)+offset];
    unsigned int mybit  = mydata&bitmask;
    // get population of ones and zeroes (cc 2.0 ballot)
    unsigned int ones = __ballot(mybit); // cc 2.0
    unsigned int zeroes = ~ones;
    offset ^= WSIZE; // switch ping-pong buffers
    // do zeroes, then ones
    if (!mybit) // threads with a zero bit
      // get my position in ping-pong buffer
      mypos = __popc(zeroes&thrmask);
    else        // threads with a one bit
      // get my position in ping-pong buffer
      mypos = __popc(zeroes)+__popc(ones&thrmask);
    // move to buffer  (or use shfl for cc 3.0)
    sdata[mypos-1+offset] = mydata;
    // repeat for next bit
    bitmask <<= 1;
    }
  // save results to global
  ddata[threadIdx.x] = sdata[threadIdx.x+offset];
  }

int main(){

  unsigned int hdata[WSIZE];
  for (int lcount = 0; lcount < LOOPS; lcount++){
    unsigned int range = 1U<<UPPER_BIT;
    for (int i = 0; i < WSIZE; i++) hdata[i] = rand()%range;
    cudaMemcpyToSymbol(ddata, hdata, WSIZE*sizeof(unsigned int));
    mykernel<<<1, WSIZE>>>();
    cudaMemcpyFromSymbol(hdata, ddata, WSIZE*sizeof(unsigned int));
    for (int i = 0; i < WSIZE-1; i++) if (hdata[i] > hdata[i+1]) {printf("sort error at loop %d, hdata[%d] = %d, hdata[%d] = %d\n", lcount,i, hdata[i],i+1, hdata[i+1]); return 1;}
    // printf("sorted data:\n");
    //for (int i = 0; i < WSIZE; i++) printf("%u\n", hdata[i]);
    }
  printf("Success!\n");
  return 0;
}
```
