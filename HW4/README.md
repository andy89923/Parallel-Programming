# Programming Assignment IV: MPI Programming
###### tags: `110Fall` `Parallel Programming` `HW`

#### **陳琮方 0816153**
[HackMD](https://hackmd.io/@CTFang/rJxzxKzUY)

[toc]

## Part 1: Getting Familiar with MPI Programming

### <font color="#0073FF"> Q1. MPI Hello World</font>


* #### How do you control the number of MPI processes on each node? 

```
$ mpirun -np X --map-by node ....
```
Variable ```X``` would be total number of process.
And ```map-by node``` could do load balancing on each node.

For controlling the number of MPI processes on each node, edit the ```hosts``` file to specific the available number of slots.
```
pp1 slots=2
pp2 slots=2
```

    
* #### Which functions do you use for retrieving the rank of an MPI process and the total number of processes?

```cpp=1
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
```

The variable ```world_size``` would be the total number of process.

```cpp=1
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
```

The variable ```world_rank``` would be the rank of current process.




### <font color="#0073FF"> Q2. MPI Blocking Communication & Linear Reduction Algorithm</font>

* #### Why ```MPI_Send``` and ```MPI_Recv``` are called “blocking” communication? 

This two command would wait until it successfully send or receive. No other instruction could do until they finish sending or recieving. So they are blocking communication.

* #### Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it 

![](https://i.imgur.com/tBdH4db.png)
![](https://i.imgur.com/hIgHLEo.png)




### <font color="#0073FF"> Q3. MPI Blocking Communication & Binary Tree Reduction Communication Algorithm</font>

* #### Measure the performance (execution time) of the code for 2, 4, 8, 16 MPI processes and plot it.

![](https://i.imgur.com/MrcaPyK.png)
![](https://i.imgur.com/F3WQdct.png)

(Binary Tree Reduction can't have NP = 12)

* #### How does the performance of binary tree reduction compare to the performance of linear reduction?

![](https://i.imgur.com/dMq5kug.png)


**-> Blocking Linear has better performance when NP is larger than 8. (Including run time and speedup)**

* #### Increasing the number of processes, which approach (linear/tree) is going to perform better? Why? Think about the number of messages and their costs.

The linear one would perform better. Each node only need one send after processing, but some nodes in tree need to sum up and send or recieve infomation more than once.





### <font color="#0073FF"> Q4. MPI Non-Blocking Communication & Linear Reduction Algorithm</font>

* #### Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

![](https://i.imgur.com/entzs4k.png)
![](https://i.imgur.com/OCn5zzT.png)



* #### What are the MPI functions for non-blocking communication?

```
MPI_Isend()
MPI_Irecv()
MPI_Wait()
MPI_Waitany()
MPI_Test()
MPI_Testany()
```

* #### How the performance of non-blocking communication compares to the performance of blocking communication?

![](https://i.imgur.com/pXzuFaq.png)

**The run time performance is blocking slower than non-blocking, but the speed up is opposite.**




### <font color="#0073FF"> Q5. MPI Collective: MPI_Gather</font>

* #### Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it. 

![](https://i.imgur.com/aNK3xXa.png)
![](https://i.imgur.com/op2W5fL.png)



### <font color="#0073FF"> Q6. MPI Collective: MPI_Reduce</font>

* #### Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

![](https://i.imgur.com/psBCzt9.png)
![](https://i.imgur.com/wllrMyD.png)




### <font color="#0073FF"> Q7. MPI Windows and One-Sided Communication & Linear Reduction Algorithm</font>

[Reference link](http://www.math.nsysu.edu.tw/~lam/mpi/PDF/chap6_slidesMY4.pdf)

* #### Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it. 

![](https://i.imgur.com/npjQYnW.png)
![](https://i.imgur.com/OjrldPy.png)


* #### Which approach gives the best performance among the 1.2.1-1.2.6 cases? What is the reason for that?

From Q3_2 and Q4_2, we could know that the **non-blocking linear** has best performance than blocking linear and binary tree reduction. 

![](https://i.imgur.com/Z97CSm1.png)

Comparing MPI Gather, MPI Reduce, and non-blocking linear, the non-blocking has the lowest running time in almost all NP. And has a better speedup when comparing with NP = 1.
-> **I think Non-linear blocking gives the best performance**. 



### <font color="#0073FF"> Q8. Measuring Bandwidth and Latency on NYCU-PP workstations with Ping-Pong</font>

[Reference video](https://www.youtube.com/watch?v=1J1aURqnwt4)

* #### Plot ping-pong time in function of the message size for cases 1 and 2, respectively.
* #### Calculate the bandwidth and latency for cases 1 and 2, respectively.


##### Case 1: intra-node communication (two processes on the same node)

```hosts``` file:
```
pp2 slots=2
```

Result:
```
         8          0.000001736
        16          0.000001896
        32          0.000001231
        64          0.000001628
       128          0.000001991
       256          0.000001944
       512          0.000002327
      1024          0.000003186
      2048          0.000003582
      4096          0.000009558
      8192          0.000013945
     16384          0.000015388
     32768          0.000020340
     65536          0.000030908
    131072          0.000043762
    262144          0.000053771
    524288          0.000068488
   1048576          0.000107983
   2097152          0.000170055
   4194304          0.000562181
   8388608          0.001422253
  16777216          0.002955408
  33554432          0.005602537
  67108864          0.010868553
 134217728          0.021631230
 268435456          0.047080842
 536870912          0.083040618
1073741824          0.165696189
```

In Matlab:
```
>> p = polyfit(siz, tim, 1)

p =

   1.0e-03 *

   0.000000155180087   0.220244002748576
```
##### bandwidth = p(2) / 1E-6 / 1E-6 (ms) = 155.18 (ms)
##### latency = 1 / p(1) / 1e9 (GB/s) = 6.44 (GB/s)

![](https://i.imgur.com/cml9UpX.png)


##### Case 2: inter-node communication  (two processes on different nodes)

```hosts``` file:
```
pp3 slots=1
pp4 slots=1
```

Result:
```
         8          0.000089663
        16          0.000069905
        32          0.000072569
        64          0.000080301
       128          0.000072117
       256          0.000068673
       512          0.000079812
      1024          0.000102523
      2048          0.000109350
      4096          0.000151703
      8192          0.000163934
     16384          0.000234883
     32768          0.000457950
     65536          0.000990574
    131072          0.001501080
    262144          0.002617184
    524288          0.004860344
   1048576          0.009356536
   2097152          0.018465014
   4194304          0.036603394
   8388608          0.072598206
  16777216          0.144676834
  33554432          0.288731635
  67108864          0.576696791
 134217728          1.152943322
 268435456          2.305589403
 536870912          4.609839647
1073741824          9.218538000
```

In Matlab:
```
>> p = polyfit(siz, tim, 1)

p =

   1.0e-03 *

   0.000008585475043   0.306933972178578
```
##### bandwidth = p(2) / 1E-6 / 1E-6 (ms) = 8585.47 (ms)
##### latency = 1 / p(1) / 1e9 (GB/s) = 0.1164 (GB/s)

![](https://i.imgur.com/R90Amh1.png)



## Part 2: Matrix Multiplication with MPI

### <font color="#0073FF"> Q9</font>
* #### Describe what approach(es) were used in your MPI matrix multiplication for each data set.

The main idea of my approach is divide the word to each process. Take NP(number of processes) = 2 for example, divide A to upper part and lower part, and each process calculate one of the part and use ```MPI_Reduce``` to reduce to process which rank = 0 (Main process), for output the answer. Also, main process(rank = 0) uses ```MPI_Bcast``` to share all the input to every process. 

##### There are no difference when dealing each dataset. (Which means that I use same approatch for each data set)
