# HuffmanCoder
Python implementation of Huffman's optimal symbol code compression algorithm.

Huffman's optimal symbol code compression algorithm, or Huffman Coding, is a data compression algorithm that assigns binary prefix codes to each symbol to be compressed, in which frequently seen symbols have shorter prefix codes than symbols that are infrequently seen.

This implementation of the algorithm dumps the constructed Huffman Code Tree (the tree in which leaves are symbols and in which prefix codes for symbols are generated) into the compressed file, which adds extra overhead, but has the nice benefit of not needing to know the original tree used to compress the file beforehand; it can reconstruct the tree from its dump.

## Usage

To compress:

```
$ ./huffman.py -e -i <file to encode> -o <compressed file>
```

To uncompress:

```
$ ./huffman.py -d -i <compressed file> -o <uncompressed file>
```



