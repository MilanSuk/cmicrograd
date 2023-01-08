
# cmicrograd
cmicrograd is full C implementation of Andrej Karpathy's(https://twitter.com/karpathy) micrograd(https://github.com/karpathy/micrograd).

This is my first interaction with neural networks. I don't speak Python very well, but luckily whole code is explained in the video "The spelled-out intro to neural networks and backpropagation- building micrograd"(https://www.youtube.com/watch?v=VMj-3S1tku0) and notebooks(https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd).

Thanks, Andrej, I learned a lot!



## Current state
- All tests passed
- under 1K LOC
- Linux only



## TODO
- Windows OS support
- read/write parameters on disc
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