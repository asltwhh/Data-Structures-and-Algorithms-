## 1. 排序

 ### 1.1 快速排序

快速排序：针对解决第k个最大或者最小的问题

```
var swap = function(arr,L,R){
    let temp = arr[L];
    arr[L] = arr[R];
    arr[R] = temp;
}
// partition负责将arr(L:R)中按照pivot分为三部分 小于pivot pivot 大于pivot
// 实现一次快排
var partition = function(arr,L,R){
    let pivotIndex = L,pivot=arr[pivotIndex];
    while(L<R){
    	// 注意：这里先检查右边，再检查左边
        while(L<R && arr[R]>=pivot){
            R--;
        }
        while(L<R && arr[L]<=pivot){
            L++;
        }
        swap(arr,L,R);
    }
    swap(arr,L,pivotIndex);
    return L;
}
// 递归排序
var quick = function(arr,L,R){
    if(L>R){
        return ;
    }
    let pivotIndex = partition(arr,L,R);
    quick(arr,L,pivotIndex-1);
    quick(arr,pivotIndex+1,R);
}
// 启动排序
var QuickSort = function(arr){
    quick(arr,0,arr.length-1);
    return arr;
}
```

对于前k个最小的问题：如果不要求前k个最小值的顺序，则直接找到第k个最小的值所在的索引，返回arr.slice(0,k)即可；如果要求前k个最小值的顺序，则在找到第k个最小的值所在的索引后，就只搜寻(L,k-1)范围内即可

```
var swap = function(arr,L,R){
    let temp = arr[L];
    arr[L] = arr[R];
    arr[R] = temp;
}
// partition负责将arr(L:R)中按照pivot分为三部分 小于pivot pivot 大于pivot
// 实现一次快排
var partition = function(arr,L,R){
    let pivotIndex = L,pivot=arr[pivotIndex];
    while(L<R){
        while(L<R && arr[R]>=pivot){
            R--;
        }
        while(L<R && arr[L]<=pivot){
            L++;
        }
        swap(arr,L,R);
    }
    swap(arr,L,pivotIndex);
    return L;
}
// 递归排序
var quick = function(arr,L,R){
    if(L>R){
        return ;
    }
    let pivotIndex = partition(arr,L,R);
    if(pivotIndex===k){
    	quick(arr,L,pivotIndex-1);
    }else{
    	quick(arr,L,pivotIndex-1);
    	quick(arr,pivotIndex+1,R);
    }
}
// 启动排序
var QuickSort = function(arr){
    quick(arr,0,arr.length-1);
    return arr;
}
```

## 2 动态规划

#### [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

**示例1:**

```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

要求时间复杂度：O(n)

思路：观测之前的结果对当前和有无贡献，如果有，则a=a+nums[i],没有则a=nums[i]
	max负责记录最大值
```

```
/**
 * @param {number[]} nums
 * @return {number}
 */
var maxSubArray = function(nums) {
    // 动态规划
    var a = nums[0]>0 ? nums[0] :0,max=nums[0];
    for(let i=1;i<nums.length;i++){
        // a  只选择a  之前+a
        a = nums[i]>a+nums[i] ? nums[i] : a+nums[i];
        // max  之前的最大值  a
        max = max>a ? max : a; 
    }
    return max;
};
```

#### [剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的**左上角**开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的**右下角**。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？   中等难度

```
var maxValue = function(grid) {
    let m=grid.length,n=grid[0].length;
    let dp = Array.from(Array(m),()=>Array(n).fill(0));
    dp[0][0] = grid[0][0];
    for(let i=1;i<m;i++){dp[i][0] = dp[i-1][0] + grid[i][0];}
    for(let j=1;j<n;j++){dp[0][j] = dp[0][j-1] + grid[0][j];}
    for(let i=1;i<m;i++){
        for(let j=1;j<n;j++){
            //  上   左
            dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1])+grid[i][j];
        }
    }
    return dp[m-1][n-1];
};
```

#### [剑指 Offer 49. 丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。

dp[i]表示第i+1个丑数。某个丑数一定是因子2 3 5乘一个小丑数产生的。记录前三个分别乘以2 3 5小于当前丑数的小丑数们，则新的丑数就是这三个小丑数乘以2 3 5得到的较小的那个

```
var nthUglyNumber = function(n) {
    if(n<=0){return -1}
    let dp = Array(n).fill(0);
    dp[0] = 1;
    let id2=0,id3=0,id5=0;
    for(let i=1;i<n;i++){
    	// 第i+1个丑数就是：大于前一个数的最小的丑数
        dp[i] = Math.min(dp[id2] * 2, Math.min(dp[id3] *3, dp[id5] * 5));
        if(dp[i] === dp[id2]*2){
            id2++;
        }
        if(dp[i] === dp[id3]*3){
            id3++;
        }
        if(dp[i] === dp[id5]*5){
            id5++;
        }
    }
    console.log(dp)
    return dp[n-1]
};
```

## 3 dfs

#### [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

难度中等

```
var translateNum = function (num) {
  var str = num.toString();
  var dfs = function (index) {
    if (index >= str.length-1) {
    //   console.log("path", path);
      return 1;
    }
    // 判断接下来的两个数字
    let temp = Number(str[index]+str[index+1]);
    if (temp < 26 && temp >= 10) {
    	// 符合条件，则分两步
        return dfs(index+1)+dfs(index+2);
    }else{
    	// 只能一步走
        return dfs(index+1);
    }
  };
  return dfs(0);
};
```

## 4 双指针

#### [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

普通解法：

```
var lengthOfLongestSubstring = function (s) {
    var m = ''
    var res = 0
    for (var i = 0; i < s.length; i++) {
    	// 如果m中不存在s[i]，则将s[i]放入m中
        if (m.indexOf(s[i]) == -1) {
            m += s[i]
        } else {
        	// 如果m中已经存在s[i],则先记录res的长度，然后重新计算m,去除原来的s[i]及其之前的元素，产生包含新的s[i]的m
            res = res < m.length ? m.length : res  // 保存长度
            m += s[i]
            m = m.slice(m.indexOf(s[i]) + 1)
        }
    }
    res = res < m.length ? m.length : res
    return res || s.length
};
```

双指针解法：

```
var lengthOfLongestSubstring = function (s) {
  let res = 0,
    l = -1;
  let map = new Map();   // 保存不重复元素及其索引
  for (let r = 0; r < s.length; r++) {
  	// 如果s[r]已经存在了，则将l移动到不包含前一个s[r]的索引处，保证l,r之间的元素不重复
    if (map.has(s[r])) {
      l = Math.max(l, map.get(s[r]));
    }
    // 更新map中s[r]的索引
    map.set(s[r], r);
    // 保存l,r之间的元素的数目，最大值就是最大不重复子串的长度
    res = Math.max(r - l, res);
  }
  return res;
};
```

