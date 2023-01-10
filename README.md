
# cmicrograd
cmicrograd is C implementation of Andrej Karpathy's([Twitter](https://twitter.com/karpathy)) micrograd([GitHub repo](https://github.com/karpathy/micrograd)).


This is my first interaction with neural networks. The original Python code is well explained in the video ["The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0) and [notebooks](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd). Thanks, Andrej, I learned a lot!


More visual explanations of what is a neural network, gradient descent and how backpropagation works can be found in [this series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).


The training loop is free of malloc() and free() for speed. The 'topo_mt.h' executes NN in multiple threads.



## Current state
- All tests passed
- under 1K LOC
- Linux only



## TODO
- Windows OS support
- read/write parameters on disk
- add example - https://github.com/karpathy/micrograd/blob/master/demo.ipynb
- more examples(recognizing handwritten numbers/letters, etc.)



## Compile & Run
<pre><code>git clone https://github.com/milansuk/cmicrograd
cd cmicrograd/linux
sh build_r
./cmicrograd_r
</code></pre>



## Repository
- /src
    - main.c - runs examples
    - examples.h
    - value.h - Value is node in neural netowrk
    - mlp.h - MLP neural network
    - topo.h - orders Values for execution(training)
    - topo_mt.h - executes Values in multiple threads
    - std.h - bridge to operation systems
- /linux - compile/run/debug scripts for Linux OS



## Author
Milan Suk

Email: milan@skyalt.com

Twitter: https://twitter.com/milansuk/



## Contributing
Your feedback and code are welcome!

For bug reports, please use GitHub's issues.

cmicrograd is licensed under **Apache v2.0** license.