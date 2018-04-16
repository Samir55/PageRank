# Search Engine PageRank

Search Page Rank based on the following article ["How Google Finds Your Needle in the Web's Haystack" by David Austin, Grand Valley State University](http://www.ams.org/publicoutreach/feature-column/fcarc-pagerank)

# Comparison between the 3 implementations
* Run on GTX 1060 gpu
* Run on Core i5 dual core processor

![](https://i.imgur.com/q7Rf98L.jpg?1)

  * Cuda (Fast implementation) is found in "/CUDA" folder
  * Cuda (Matrix Slow implementation) is found in "/CUDA(Matrices)" folder
  * Java optimized code is found in "Java/src" folder

# Todo:
* Add Machine learning technique beside PageRank.

# Build
```Console
cd CUDA
mkdir build
cd build
cmake ..
make 
./PageRank the_input_edges_list
```
