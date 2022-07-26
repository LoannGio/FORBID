# Compile

This source code has been tested with C++17 and does not use any third party library. 

To compile on Windows or Linux, run the following command in a bash or powershell: 

```$ g++ -std=c++17 -funroll-loops -O3 FORBID.cpp randomkit.c -o FORBID.exe```

# Run

To run the code, you must provide all the parameters expected by the code. The description of all parameters is given below. But first, here follows an example command with some default values for all parameters. 

```$ ./FORBID.exe <INPUT_LAYOUT_PATH> <ALPHA> <K> <MINIMUM_MOVEMENT> <MAX_ITER> <MAX_PASSES> <SCALE_STEP> <PRIME> ```

| Parameter  |  Type |  Description |
|:---:|:---:|:---:|
| ``INPUT_LAYOUT_PATH``  | string  |  path to the .txt file encoding the initial graph layout in the expected format (described below). The algorithm will output the result layout in ``INPUT_LAYOUT_PATH.forbid`` or ``INPUT_LAYOUT_PATH.forbidp`` depending on the ``PRIME`` argument |
|  ``ALPHA`` | float |  weight factor for ideal distance for both overlapped and non-overlapped pairs of nodes |
|  ``K`` | float  | additional weight factor for overlapped pairs of nodes  |
|  ``MINIMUM_MOVEMENT`` |  float |  threshold value for the optimization algorithm. In a pass in the optimization algorithm, if the sum of nodes movement is below that threshold, the pass is ended. |
|  ``MAX_ITER`` |  int |  maximum number of iterations in each pass in the optimization algorithm |
|  ``MAX_PASSES`` |  int |  maximum number of passes before existing FORBID (should not be reached)  |
|  ``SCALE_STEP`` | float  | minimal step size that stops the binary search for the optimal scale  |
|  ``PRIME`` | bool as int (0,1)  | wether to use the FORBID or FORBID' variant  |

Example: 

```$ ./FORBID.exe ../demo/datasets_txt/graphviz/mode.txt 2 4 0.00001 30 50 0.1 1 ```

# Input graph format

The input graph should be a txt file in which:
* the first line contains the graph number of nodes (i.e., number of lines to parse)
* one line per node in the format ``x y width height`` as integer of float values.

Example for a small graph with 4 nodes:
```
4
1 2 3 3
10 10 2.5 2.5
7 7 12 1.5
0 9 11 96
```

# Output graph format

The algorithm will output a file in almost the same format as its input, except that the first line is the execution time of the algorithm. The algorithm result is stored in a new file created at the same location as the input file, with the same name on which is appended the extension ``.res``