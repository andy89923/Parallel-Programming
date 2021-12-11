# Programming Assignment V: CUDA Programming
###### tags: `110Fall` `Parallel Programming` `HW`

#### **陳琮方 0816153**
[HackMD](https://hackmd.io/@CTFang/ByHR6t_YY)

[toc]


## <font color="#0073FF"> Q1. Pros & Cons of the three methods</font>
* ### Each CUDA thread processes one pixel -- With ```malloc``` & ```cudaMalloc```
#### Pros:

每個 thread 只處理一個 pixel，效率不會受到分配不均的影響 

#### Cons:

一次就會需要非常多的 thread ，且有些用了一下下就算完了，造成 thread utilization 較低


* ### Each CUDA thread processes one pixel -- With ```cudaHostAlloc``` & ```cudaMallocPitch```
#### Pros:

每個 thread 只處理一個 pixel，效率不會受到分配不均的影響 
且在使用資料為二維的時候，可以讓資料放在一起（對齊），讓 cuda 一次讀取時比較快。

#### Cons:

因為也是一個 thread 計算一個 pixel，所以也會有 thread utilization 較低的問題。
並且如果資料量不是 cuda 一次讀取的大小，會需要額外的 overhead 去處理。


* ### Each CUDA thread processes a group of pixels -- With ```cudaHostAlloc``` & ```cudaMallocPitch```
#### Pros:

每個 thread 計算較多的 pixel ，所以需要的計算資源較低

#### Cons:

會有資料分配不平均的狀況，某些需計算多的聚在一起，被分配到的 thread 就要一次計算很多



## <font color="#0073FF"> Q2. Performances of the three methods</font>

### Performances: Method 1 > Method 2 > method 3

![](https://i.imgur.com/9ctfUiX.png)
![](https://i.imgur.com/prpqjUN.png)
![](https://i.imgur.com/Rn9r64t.png)



## <font color="#0073FF"> Q3. Explain the performance differences</font>

* Method 1 and 2 比 Method 3 快

    * 因為 Method 1 & 2 都是一個 thread 跑一個 pixel，所以不會有 Method 3 計算量分佈不平均之問題
    * Method 3 因為一個 thread 負責多個 pixel ，所以需計算多的 pixel 剛好被分配到同一個 thread ，那這個 thread 就相較其他 thread ，會需要更多的計算時間，效率較差
    * View 1 跟 View 2 都可以明顯看到上述情況，在 HW2 也有分析過兩張的白色區域(計算量大的地方)分佈，造成兩張差距盛大的結果

* Method 1 < Method 2

    * 這兩個的效率理論上 Method 2 會比較快，因為使用 ```cudaNallocPitch``` ，在讀取二維資料的時候因為有 align，應該要比較快，但實驗結果不然
    * 推測原因為本次實驗需要讀取的記憶體不多，故效果不顯著，帶來的好處不足以彌補 ailgn 的 overhead (且要使用```cudaDeviceSynchronize()```)
    * 在方法上因為都是 one pixel per thread ，所以在 MaxIterations = 100000 的狀況下幾乎有一樣的 Runtime




## <font color="#0073FF"> Q4. Can we do even better?</font>

我發現並不需要使用 ```malloc``` 或是 ```cudaHostAlloc``` 先開一塊空間，將計算結果搬到記憶體，再用 ```memcpy```放到 img 上，我們可以直接將計算結果搬到 img 上，省下一點點的記憶體複製的時間。

![](https://i.imgur.com/lQkrx4M.png)
![](https://i.imgur.com/uHACO4H.png)
![](https://i.imgur.com/qPP2Rcw.png)

從實驗結果可以看到，在不同的測試當中，Method 4 幾乎都有最快速的 runtime。

---

You could see chart data on [Google Sheets](https://docs.google.com/spreadsheets/d/12RNlfsA9rI-JYkPXUSYn01RQOW5QyHatONdJZaLXET8/edit?usp=sharing).
