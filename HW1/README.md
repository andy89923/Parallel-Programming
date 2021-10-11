# Programming Assignment I: SIMD Programming

###### tags: `110Fall` `Parallel Programming` `HW`

 **陳琮方 0816153**
[HackMD](https://hackmd.io/@CTFang/BkbL7K4VF)

[toc]

## Q1 
###### Run ./myexp -s 10000 and sweep the vector width from 2, 4, 8, to 16. Record the resulting vector utilization. You can do this by changing the #define VECTOR_WIDTH value in def.h. Q1-1: Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

First, we change the vecto width in `def.h` to target width, and test for the result.

* **Vector width = 2**
![](https://i.imgur.com/KsywkKF.png)

* **Vector width = 4**
![](https://i.imgur.com/N0OJDuL.png)

* **Vector width = 8**
![](https://i.imgur.com/HSta4DF.png)

* **Vector width = 16**
![](https://i.imgur.com/R8HfRiA.png)

### Conclution
* The vector utilization is decrease as the vector width increase.
* When the vector width increase, there are more elements in the vector. 
* Every element must wait for all other element s are done.
* More elements, higher chance to wait
* Vector utilization decrease


### Bonus - arraySumVector

![](https://i.imgur.com/imB5wlF.png)

![](https://i.imgur.com/hkpyg0N.png)



## Q2-1 Make sure use Aligned moves

AVX2 expands most integer commands to 256 bits which is 32 bytes.
So I change the code:
```cpp=1
a = __builtin_assume_aligned(a, 16)
b = __builtin_assume_aligned(b, 16)
c = __builtin_assume_aligned(c, 16)
```
to 32 bytes aligned
```cpp=1
a = __builtin_assume_aligned(a, 32)
b = __builtin_assume_aligned(b, 32)
c = __builtin_assume_aligned(c, 32)
```
Check assembly is `vmovaps` rather than `vmovups`.
![](https://i.imgur.com/xGtSgp6.png)

Reference
[Wiki - Advanced Vector Extensions](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)

## Q2-2 
###### What speedup does the vectorized code achieve over the unvectorized code? What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)? You may wish to run this experiment several times and take median elapsed times; you can report answers to the nearest 100% (e.g., 2×, 3×, etc). What can you infer about the bit width of the default vector registers on the PP machines? What about the bit width of the AVX2 vector registers.

###### Hint: Aside from speedup and the vectorization report, the most relevant information is that the data type for each array is float.

* **make clean && make && ./test_auto_vectorize -t 1**

![](https://i.imgur.com/fBd4N0o.png)
Ave. Time ≈ 8.3 sec

* **make clean && make VECTORIZE=1 && ./test_auto_vectorize -t 1**

![](https://i.imgur.com/4ABdwKg.png)
Ave. Time ≈ 2.6 sec

* **make clean && make VECTORIZE=1 AVX2=1 && ./test_auto_vectorize -t 1**

![](https://i.imgur.com/GOGf78L.png)
Ave. Time ≈ 1.4 sec

### Conclution
Compare serial with vectorize, vectorize code would move variables to vector using one load data time. And do the calculation at the same time. So vectorize version is much faster than serial version. 

Changing AVX to AVX2, the width of registers are two times larger. As the result, the time usage is approximately half of time usage of AVX version.
* Floating Point registers
    *  AVX : XMM (128 bits, 4 float numbers)
    *  AVX2: YMM (256 bits, 8 float numbers)


## Q2-3
###### Provide a theory for why the compiler is generating dramatically different assembly.

**Version 1**
```c++=1
c[j] = a[j];
if (b[j] > a[j]) c[j] = b[j];
```

![](https://i.imgur.com/zXUUMwM.png)
This version would asssign value (Line 62) at first, and compare the values(Line 64 ucomiss).  There are no vectorization in this version.

**Version 2**
```c++=1
if (b[j] > a[j]) c[j] = b[j];
else c[j] = a[j];
```

![](https://i.imgur.com/LX8oWat.png)

Compare with the previous version, the compiler assign `maxps` command to the code which is SIMD function.

### Conclution
This two version of code give compiler different look.

**Version 1**

```flow
st=>start: START
ed=>end: END
cod=>condition: IF
op1=>operation: Assign A
op2=>operation: Assign B

st->op1->cod
cod(yes)->op2->ed
cod(no)->ed
```

**Version 2**

```flow
st=>start: START
ed=>end: END
cod=>condition: IF
op1=>operation: Assign A
op2=>operation: Assign B

st->cod
cod(yes)->op2->ed
cod(no)->op1->ed
```
The main different between these two version is when to assign values. The version 1 would assign values first, and makes it can't parallelize. The version 2 is opposite, there is `maxps` which is in **SIMD comparison function**. Lead to two different assembly but doing the same things.