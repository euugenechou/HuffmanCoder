#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A Python 2 implementation of the Huffman Coding algorithm.

Author: Eugene Chou
Email: euchou@ucsc.edu
Last Revised: 3/13/19
"""

from Queue import PriorityQueue
import sys
import argparse
import array
import collections
import os


class BitVector(object):
    """
    A BitVector class to streamline reading/writing of bits from files.
    Modified from: engineering.purdue.edu/kak/dist/BitVector-1.0.html
    """

    def __init__(self, arg):
        """
        The constructor for a BitVector.

        Attributes:
            size (int): The size or length of a BitVector
            filename (str): The filename a specified file.
            inpfile (file): The opened file.
            more_to_read (bool): True if the input file has unread bits.
            vector (array): The bit array (basically the BitVector)

        Parameters:
            arg: can be either an int, list, tuple, or string.
            If arg is a int, an empty BitVector of size arg is created.
            If arg is a list or tuple, a BitVector is created from the
            1's and 0's in the list or tuple.
            If arg is a string, a BitVector is created with an input
            file which is opened.
        """

        if isinstance(arg, (int)):
            self.size = arg
            bits = [0] * arg
        elif isinstance(arg, (list, tuple)):
            self.size = len(arg)
            bits = arg
        elif isinstance(arg, (str)):
            self.filename = arg
            self.inpfile = open(arg, 'rb')
            self.size = 0
            self.more_to_read = True
            return

        two_byte_ints_needed = (len(bits) + 15) // 16
        self.vector = array.array('H', [0] * two_byte_ints_needed)

        if isinstance(arg, (int)):
            map(self.setbit, enumerate(bits), bits)
        else:
            map(self.setbit, enumerate(bits), arg)

    def setbit(self, pos, val):
        """
        Sets a bit in the BitVector at a certain position.

        Parameters:
            pos (int): Where to set the bit.
            val (int): What value to set the bit to.
        """

        if val not in (0, 1):
            raise ValueError("incorrect value for a bit")

        if isinstance(pos, (tuple)):
            pos = pos[0]

        if pos >= self.size:
            raise ValueError("index range error")

        index = pos // 16
        shift = pos & 15
        cv = self.vector[index]

        if (cv >> shift) & 1 != val:
            self.vector[index] = cv ^ (1 << shift)

    def getbit(self, pos):
        """
        Gets a bit in the BitVector at a certain position.

        Parameters:
            pos (int): Where to get the bit at.

        Returns:
            val (int): The value of the specified bit.
        """

        if pos >= self.size:
            raise ValueError("index range error")

        val = (self.vector[pos//16] >> (pos & 15)) & 1
        return val

    def getsize(self):
        """
        Gets the size or length of the BitVector.
        """

        return self.size

    def bitstring(self):
        """
        Constructs and returns a bitstring from the BitVector.

        Returns:
            bitstring (str): The BitVector's string representation.
        """

        bitstring = ""

        for i in range(self.size):
            bitstring += str(self.getbit(i))

        return bitstring

    def readblock(self, blocksize):
        """
        Reads and converts a block into a bits from the BitVector's
        input file

        For each byte read, take its hex value and get its binary
        string. The binary will then be appended to a bitstring and
        returned.

        Parameters:
            blocksize (int): The size of the block to read.

        Returns:
            bitstring (str): The string representation of the read block.
        """

        # hex dictionary for conversion hex into a binary string
        hexdict = {
            '0':"0000", '1':"0001", '2':"0010", '3':"0011",
            '4':"0100", '5':"0101", '6':"0110", '7':"0111",
            '8':"1000", '9':"1001", 'a':"1010", 'b':"1011",
            'c':"1100", 'd':"1101", 'e':"1110", 'f':"1111"
        }

        bitstring = ""

        for _ in range(blocksize / 8):
            byte = self.inpfile.read(1)

            if not byte:
                return bitstring

            hexvalue = hex(ord(byte))
            hexvalue = hexvalue[2:]

            if len(hexvalue) == 1:
                hexvalue = "0" + hexvalue

            bitstring += hexdict[hexvalue[0]]
            bitstring += hexdict[hexvalue[1]]

        return bitstring

    def read_bits_from_file(self, blocksize):
        """
        Reads block size number of bits from the BitVector's input file.

        If there are no bits in the file to read, set more_to_read to
        False. Calls readblock() to get the bitstring, which is used to
        create and return a BitVector.

        Parameters:
            blocksize (int): The size of the block (number of bits).

        Returns:
            bitvec (BitVector): The BitVector containing the read bits.
        """

        if not self.filename:
            raise ValueError("input file required!")

        bitstring = self.readblock(blocksize)

        if len(bitstring) < blocksize:
            self.more_to_read = False

        bitlist = map(lambda x: int(x), list(bitstring))
        bitvec = BitVector(bitlist)
        return bitvec

    def write_to_file(self, outfile):
        """
        Writes the bits in the BitVector to a file.

        Warning: the size of the BitVector must be a multiple of 8.
        Iterates through the BitVector and writes the value of every
        8 bits since the smallest size Python can write is a byte.

        Parameters:
            outfile (file): The specified output file to output bits to.
        """

        for byte in range(self.size / 8):
            value = 0

            for bit in range(8):
                value += (self.getbit(byte * 8 + (7 - bit)) << bit)

            outfile.write(chr(value))

    # these dunder functions allow list-like access for BitVectors
    __getitem__ = getbit
    __setitem__ = setbit
    __len__ = getsize


class PriorityNodeQueue(PriorityQueue):
    """
    A class that acts as a wrapper for the Python's PriorityQueue.
    Written since the default PriorityQueue doesn't support push/pop.
    Directly inherits from PriorityQueue.

    Attributes:
        counter (int): The number of nodes in the queue.
    """

    def __init__(self):
        """
        The constructor for the PriorityNodeQueue class.
        """

        PriorityQueue.__init__(self)
        self.counter = 0

    def push(self, node, priority):
        """
        Pushes a node into the queue.

        Parameters:
            node (CodeTreeNode): The node to be pushed into the queue.
            priority (int): The priority of the node to be pushed.
        """

        PriorityQueue.put(self, (priority, self.counter, node))
        self.counter += 1

    def pop(self):
        """
        Pops the node with the lowest frequency from the queue.

        Returns:
            node (CodeTreeNode): The node with the lowest frequency.
        """

        _, _, node = PriorityQueue.get(self)
        return node

    def empty(self):
        """
        Checks whether or not the queue is empty.

        Returns:
            empty (bool): Whether or not the queue is empty.
        """

        empty = PriorityQueue.empty(self)
        return empty


class CodeTreeNode(object):
    """
    A modified node class to construct the Huffman Code Tree with.

    Attributes:
        symbol (chr): The node's symbol.
        count (int): The frequency in which the symbol appears.
        is_leaf (bool): Whether or not the node is a leaf.
        left (CodeTreeNode): The node's left child
        right (CodeTreeNode): The node's right child
    """

    def __init__(self, symbol, count, is_leaf, left, right):
        """
        The constructor for the CodeTreeNode class

        Parameters:
            symbol (chr): The node's symbol.
            count (int): The frequency in which the symbol appears.
            is_leaf (bool): Whether or not the node is a leaf.
            left (CodeTreeNode): The node's left child
            right (CodeTreeNode): The node's right child
        """

        self.symbol = symbol
        self.count = count
        self.is_leaf = is_leaf
        self.left = left
        self.right = right
        self.code = ""


##### COMPRESSION FUNCTIONS #####

def create_codes(node, curr_code, codes):
    """
    Recursive function to generate all prefix codes given the root
    of the Huffman Tree.

    "0" is appended to the current code when going down the left path,
    and "1" is appended when going down the right path. When a leaf is
    found, set the code for the leaf node's symbol as the current code.

    Parameters:
        node (CodeTreeNode): The current node in the Huffman tree.
        curr_code (str): The current prefix code.
        codes (dict): The dictionary of all codes for each symbol.
    """

    if not node:
        return

    if not node.left and not node.right:
        codes[node.symbol] = curr_code
        return

    if node.left:
        create_codes(node.left, curr_code + "0", codes)

    if node.right:
        create_codes(node.right, curr_code + "1", codes)


def create_code_tree(histogram):
    """
    Creates the Huffman Code Tree using a priority queue.

    For each unique symbol in the histogram, create and push a node for
    the symbol into the queue. The tree itself is built by repeatedly
    popping off two nodes from the queue, will be be, respectively,
    the left and right nodes. Then create a new parent node whose symbol
    is '$' and count is the sum of the left and right node counts.
    Push the parent node onto the queue and repeat. Terminate when
    the queue is empty after popping twice, and return the created
    parent node as the root of the tree.

    Parameters:
        histogram (Counter): A counter of symbols and their frequencies.

    Returns:
        parent (CodeTreeNode): The root of the Huffman Code Tree.
    """

    pq = PriorityNodeQueue()

    for symbol, count in histogram.iteritems():
        new_node = CodeTreeNode(symbol, count, True, None, None)
        pq.push(new_node, count)

    while not pq.empty():
        left = pq.pop()
        right = pq.pop()

        parent_count = left.count + right.count
        parent = CodeTreeNode('$', parent_count, False, left, right)

        if pq.empty():
            return parent

        pq.push(parent, parent_count)


def get_tree_dump(root):
    """
    Constructs the dump of the Huffman Code Tree.

    This is essentially a string of leaves and internal codes
    obtained through a post-order traversal that represents the tree.
    If a leaf is encountered, append an 'L' followed by the leaf's
    symbol to the tree dump string. Else append an 'I' to denote an
    internal node.

    Parameters:
        root (CodeTreeNode): The root of the Huffman Code Tree.

    Returns:
        tree_dump (str): The string representation of the tree.
    """

    if not root:
        return

    stack = []
    tree_dump = ""

    while True:
        while root:
            if root.right:
                stack.append(root.right)

            stack.append(root)
            root = root.left

        root = stack.pop()

        peek = stack[-1] if len(stack) > 0 else None

        if root.right and peek == root.right:
            stack.pop()
            stack.append(root)
            root = root.right
        else:
            if root.is_leaf:
                tree_dump += "L{}".format(root.symbol)
            else:
                tree_dump += "I"
            root = None

        if len(stack) <= 0:
            break

    return tree_dump


def create_histogram(inpfname):
    """
    Creates a histogram, or a dictionary of symbols and their
    frequency of appearance in the input file.

    Parameters:
        inpfname (str): The name of the input file.

    Returns:
        histogram (Counter): A counter of symbols and their frequencies.
    """

    histogram = collections.Counter()

    with open(inpfname, 'rb') as inpfile:
        while True:
            symbol = inpfile.read(1)
            if not symbol:
                break
            histogram[symbol] += 1

    return histogram


def get_tree_size(histogram):
    """
    Calculates the tree size by finding the number of leaf nodes there
    are in the tree.

    The number of leaf nodes is always the number of unique symbols
    there are in the input file. Due to how the tree is dumped, the
    size of the tree is always 3 * number of leaves -1.

    Parameters:
        histogram (Counter): A counter of symbols and their frequencies.

    Returns:
        tree_size (int): The number of nodes (or size) of the tree.
    """

    tree_size = 3 * len(histogram) - 1
    return tree_size


def compress(inpfname, outfname):
    """
    Compresses the contents of the input file into the output file.

    First, create a histogram from the symbols in the input file, then
    construct a Huffman Code Tree from the histogram and generate prefix
    codes for each symbol from the tree. After that, step through the
    input file and build up a bitstring by concatenating each symbol's
    prefix code. Lastly, write the original file size, tree size,
    tree dump, and bitstring (the actual bits, not the string itself)
    to the output file.

    Parameters:
        inpfname (str): The name of the input file.
        outfname (str): The name of the output file.
    """

    histogram = create_histogram(inpfname)

    root = create_code_tree(histogram)

    codes = {}
    create_codes(root, "", codes)

    tree_dump = get_tree_dump(root)

    tree_size = str(get_tree_size(histogram))

    code_string = ""
    with open(inpfname, 'rb') as inpfile:
        while True:
            symbol = inpfile.read(1)
            if not symbol:
                break
            code_string += codes[symbol]

    # backfill zeroes so that the bitstring ends on a byte
    if len(code_string) % 8 != 0:
        backfill_zeroes = 8 - (len(code_string) % 8)
        padding = "0" * backfill_zeroes
        code_string += padding
        assert len(code_string) % 8 == 0

    file_size = str(os.path.getsize(inpfname))

    with open(outfname, 'wb+') as outfile:
        outfile.write("%s:%s:%s" % (file_size, tree_size, tree_dump))

        # create a BitVector from code_string
        bitstring = [int(i) for i in code_string]
        bitvec = BitVector(bitstring)

        bitvec.write_to_file(outfile)



##### UNCOMPRESSION FUNCTIONS #####

def construct_tree(tree_dump):
    """
    Constructs a Huffman Code Tree given the tree dump from a
    post-order traversal.

    Iterates through the tree dump string. If an 'L' is encountered,
    the next character in the string is the symbol, so create a for
    this symbol (its frequency doesn't matter) and push it into a stack.
    Else if an 'I' is encountered, pop the stack twice for the right
    and left nodes respectively. Create a parent node whose symbol
    is '$', set its left and right nodes to the nodes that were just
    popped, then push it into the stack. The root of the tree will be
    the one node in the stack after iteration of the tree dump string
    has finished, so pop and return it.

    Parameters:
        tree_dump (str): The tree dump from post-order traversal.

    Returns:
        root (CodeTreeNode): The root of the Huffman Code Tree.
    """

    stack = []

    output_found_leaf = False
    for symbol in tree_dump:
        if output_found_leaf:
            new_node = CodeTreeNode(symbol, 0, True, None, None)
            stack.append(new_node)
            output_found_leaf = False
        else:
            if symbol == 'L':
                output_found_leaf = True
            else:
                right = stack.pop()
                left = stack.pop()
                parent = CodeTreeNode('$', 0, False, left, right)
                stack.append(parent)

    root = stack.pop()
    return root


def convert_code_to_symbols(bitstring, file_size, root):
    """
    Converts a bitstring of 1's and 0's back into symbols by stepping
    through the Huffman Code tree.

    Iterates through the bitstring. If the current node is a leaf,
    append the leaf's symbol to the uncompressed string, increment the
    number of uncompressed symbols, and reset the current node to the
    root node. Break out of the loop when the number of uncompressed
    symbols is equal to the original file size. Else, if the current
    bit in the bitstring is a '0', set the current node to its left
    child. Else if the current bit is a '1', the the current node to
    its right child. After all the symbols have been uncompressed,
    return the uncompressed string.

    Parameters:
        bitstring (str): The string of concatenated prefix codes.
        file_size (int): The size of the original file.
        root (CodeTreeNode): The root of the Huffman Code Tree

    Returns:
        uncompressed_string (str): The string of uncompressed symbols.
    """

    curr = root
    uncompressed_string = ""
    uncompressed_symbols = 0

    for bit in bitstring:
        if curr.is_leaf:
            uncompressed_string += curr.symbol
            curr = root
            uncompressed_symbols += 1

            if uncompressed_symbols == file_size:
                break

        curr = curr.left if bit == "0" else curr.right

    return uncompressed_string


def uncompress(inpfname, outfname):
    """
    Uncompresses the contents of the input file into the output file.

    Opens the specified input file, recovers the original file size,
    tree size, and tree dump. From the tree dump, reconstruct the
    original Huffman Code Tree. With the reconstructed tree, recreate
    the prefix codes for each unique symbol. Then, read in the rest of
    the bits in the input file, convert them to a bitstring, and convert
    them back into symbols by stepping through the tree. Write the
    decoded symbol string into the specified output file.

    Parameters:
        inpfname (str): The name of the input file.
        outfname (str): The name of the output file.
    """

    with open(inpfname, 'rb') as inpfile:
        # keep track of number of read bytes as the offset
        # we'll need this as an offset when we read the rest
        # of the bits in the input file
        offset = 0

        file_size_string = ""
        while True:
            symbol = inpfile.read(1)
            offset += 1
            if symbol == ':':
                break
            else:
                file_size_string += symbol
        file_size = int(file_size_string)

        tree_size_string = ""
        while True:
            symbol = inpfile.read(1)
            offset += 1
            if symbol == ':':
                break
            else:
                tree_size_string += symbol
        tree_size = int(tree_size_string)
        offset += tree_size

        tree_dump = ""
        for i in range(tree_size):
            symbol = inpfile.read(1)
            tree_dump += symbol

    root = construct_tree(tree_dump)

    codes = {}
    create_codes(root, "", codes)

    bitstring = ""
    bitvec = BitVector(inpfname)
    bitvec.inpfile.seek(offset, 0)

    # read bits from the input file in block sizes of 32 into a BitVector
    # if bits were read in, append them to the bitstring
    while bitvec.more_to_read:
        read_bits = bitvec.read_bits_from_file(32)

        if read_bits.getsize() > 0:
            bitstring += read_bits.bitstring()

    with open(outfname, 'wb+') as outfile:
        uncompressed = convert_code_to_symbols(bitstring, file_size, root)
        outfile.write(uncompressed)


if __name__ == '__main__':
    """
    The main driver for the program. Parses the commandline arguments
    for either compressing or uncompressing, and keeps track of the
    specified input and output filenames for either mode.
    """

    parser = argparse.ArgumentParser(description="Huffman Coder")
    parser.add_argument('-u', '--uncompress', help='uncompress',
        action='store_true', default=False)
    parser.add_argument('-c', '--compress', help='compress',
        action='store_true', default=False)
    parser.add_argument('-i', '--input', help='input filename',
        nargs=1, required=True)
    parser.add_argument('-o', '--output', help='output filename',
        nargs=1, required=True)
    args = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    inpfname = args.input[0]
    outfname = args.output[0]

    if args.compress:
        compress(inpfname, outfname)
    if args.uncompress:
        uncompress(inpfname, outfname)

