# Programming Assignment II: Multi-thread Programming

###### tags: `110Fall` `Parallel Programming` `HW`

#### **陳琮方 0816153**
[HackMD](https://hackmd.io/@CTFang/rJxzxKzUY)

[toc]

## Part 1: Parallel Counting PI Using Pthreads

### <font color="#0073FF"> Performance </font>
![](https://i.imgur.com/f0ivzzs.png)

#### Note
* **real** - is the time from start to finish of the call. It is the time from the moment you hit the Enter key until the moment the wget command is completed.
* **user** - amount of CPU time spent in user mode.
* **sys** - amount of CPU time spent in kernel mode.


## Part 2: Parallel Fractal Generation Using std::thread

### <font color="#0073FF"> Q1. Speedup compared to the reference sequential implementation </font>

:::info
Extend your code to use 2, 3, 4 threads, partitioning the image generation work accordingly (threads should get blocks of the image). 
Produce a graph of speedup compared to the reference sequential implementation as a function of the number of threads used FOR VIEW 1. 
Q1. Is speedup linear in the number of threads used? In your writeup hypothesize why this is (or is not) the case? 
> You may also wish to produce a graph for VIEW 2 to help you come up with a good answer.  Hint: take a careful look at the three-thread data-point.
:::

* #### View 1
![](https://i.imgur.com/oahEdsV.png)

* #### View 2

![](https://i.imgur.com/PNT9aQW.png)

* #### Line chart
> X-axis stands for the number of thread(s)
![](https://i.imgur.com/8hYlDBi.png)


#### Conclution & Hypothesize

As the result above, the speedup linear is <font color="red">**not**</font> the number of threads used.
The reason is not sure yet. However, the speedup depend on the number of computation in each thread. The distribution of number in view 1 may cause the different thread has different computaion needs (different area, different numbers). The works(computaion) of different threads are not balanced in 3 thread parallelism cause the total computaion time higher than 2 thread parallelism. 
The reason will be confirmed in Q2.



### <font color="#0073FF"> Q2. How do your measurements explain the speedup graph you previously created?</font>
:::info
To confirm (or disprove) your hypothesis, measure the amount of time each thread requires to complete its work by inserting timing code at the beginning and end of workerThreadStart(). 
Q2: How do your measurements explain the speedup graph you previously created?
:::

#### I added the timer for every thread from created to prove my hypothesize in Q1, and tested the result for view 1.

* #### Thread 2

![](https://i.imgur.com/2pv84Aa.png)

> All threads have approximately same running time. Balanced.

* #### Thread 3

![](https://i.imgur.com/5gH1ZfT.png)

> The maximum running time is approximately 3 time longer than the minimum. It show that the computaion for each thread are not balanced.

* #### Thread 4

![](https://i.imgur.com/dmt8Soj.png)

> The situation in thread 3 can also be found in thread 4. Works(computaion) for each thread are not balanced.

#### Conclution

![](https://camo.githubusercontent.com/80f2e33b4e20f3f86809c6203402dc6807b389bc/687474703a2f2f67726170686963732e7374616e666f72642e6564752f636f75727365732f6373333438762d31382d77696e7465722f617373745f696d616765732f61737374312f6d616e64656c62726f745f76697a2e6a7067)

In View 1, the area of white pixel are more in the middle than the upper and lower. So the thread 1 (in 3 thread parallelism) which distributed to calculate the middle part took much more time than the others. When we use more thread on View 1, the distribuiton of number unblanced which cause that needs more computation in specific thread are not that obvious compare to 3-thread. 

Looking at the View 2, the distribution of white pixel are much balance than View 1. So the times of speedup are more liner speedup in the number of thread use. 





### <font color="#0073FF"> Q3. Describe your approach to parallelization and report the final 4-thread speedup obtained.</font>
:::info
Modify the mapping of work to threads to achieve to improve speedup to at about 3-4x on both views of the Mandelbrot set (if you’re above 3.5x that’s fine, don’t sweat it). You may not use any synchronization between threads in your solution. We are expecting you to come up with a single work decomposition policy that will work well for all thread counts—hard coding a solution specific to each configuration is not allowed! 
(Hint: There is a very simple static assignment that will achieve this goal, and no communication/synchronization among threads is necessary.). 
:::

To avoid the similar problem above, I distributed continous row to different thread instead segment of rows. This approach can avoid the problems of distrubution of numbers that need a lot of computation nearby. This approach use the following code in the ```mandelbrotThread.cpp``` line start at 43.

```cpp=43
int t = args -> numRows;
int k = args -> startRow;
int s = args -> numThreads;
int h = (int) args -> height;
for (int i = 0; i < t; ++i) {
    mandelbrotSerial(args -> x0, args -> y0, args -> x1, args -> y1, args -> width, 
                     args -> height, k, 1, args -> maxIterations, args -> output );

    if (k + s >= h) k += 1; // out of the height range (mod paart)
    else
        k += s;
}
```


#### <font color="#0073FF">Requirement </font>

##### ```$./mandelbrot -t 3``` + ```$./mandelbrot -t 4 ```
![](https://i.imgur.com/2bCDqjw.png)
##### => 159.642 ms + 121.577 ms = 281.219 ms **<**  0.375 s = 375 ms

This speedup also met the requirement. :+1: 






### <font color="#0073FF"> Q4: Run the program with 8 threads </font>
:::info
Q4. Now run your improved code with eight threads. Is performance noticeably greater than when running with four threads? Why or why not? 
(Notice that the workstation server provides 4 cores 4 threads.)
:::

I used the following command to test the performance:
##### ```$./mandelbrot -v 1 -t k``` for k = 1 ~ 8

![](https://i.imgur.com/s3TeIZY.png)

The result shows that there seem to be a bottleneck at 4 thread.
> Reason: Workstation server provides 4 cores 4 threads

Computer just have 4 CPU with total 4 threads, so there is <font color="red">no way</font> that 5 or above (includeing 8) threads running at the same time. As the result, the bottleneck of speedup is at 4 thread with approximately 3.78x speedup.