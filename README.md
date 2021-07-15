## 数量级与时间复杂度

![](./img/01.png)

一般leetcode给出的超时限制是1s,上面给出的应该是1s内各个复杂度可以处理的数量级。比如，当n是10^4数量级时，可以接受O(nlogn）、O(n)等复杂度的算法。O(n^2)的算法就可能会超时了

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
// 递归
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

#### [剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

难度中等245

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

示例 1:

输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
示例 2:

输入: 2
输出: [0.02778,0.05556,0.08333,0.11111,0.13889,0.16667,0.13889,0.11111,0.08333,0.05556,0.02778]

分析：

1个骰子，得到的结果：1~6

2个骰子，得到的结果：2~12     n~n*6

`dp[i][j]`表示i个骰子得到和为j的概率，则可以推出：

`dp[3][3] = dp[1][1]*dp[2][2] + dp[1][2]*dp[2][1] + dp[1][3]*dp[2][0];`

```
 var dicesProbability = function(n) {
    let dp = Array.from(Array(n+1),()=>Array(6*n+1).fill(0));
    // 初始化，一个骰子掷出1~6的概率均为1/6，2个及以上骰子掷出0或者1的概率为0
    for(let i=1;i<7;i++){
        dp[1][i] = 1/6.0;
    }    
    // 从 至少两个骰子掷出2开始
    for(let i=2;i<n+1;i++){
        // i个骰子
        for(let j=i;j<6*i+1;j++){
            // 骰出几 
            for(let k=1;k<7;k++){
                if(i-k>=0 && j-k>=0){
                    // i个骰子和为j  1个骰子为k  i-1个骰子和为j-k
                    dp[i][j] += dp[1][k]*dp[i-1][j-k];
                }
            }
        }
    }
    // 记录所有结果的值
    let result = Array(5*n+1).fill(0);
    let index = 0;
    // n个骰子得到的结果范围： n~6*n
    for(let i=n;i<6*n+1;i++){
        result[index++] = dp[n][i];
    }
    return result;
};
```

#### [343. 整数拆分](https://leetcode-cn.com/problems/integer-break/)

难度中等527

给定一个正整数 *n*，将其拆分为**至少**两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

**示例 1:**

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。
```

分析：dp[i]表示整数i拆分为两个整数后得到的最大乘积  `

```
var integerBreak = function(n) {
    var dp = Array(n+1).fill(0);
    // console.log(dp)
    for(let i=2;i<n+1;i++){
        for(let k=1;k<i;k++){
            dp[i] = Math.max(dp[i],k*dp[i-k],k*(i-k));
        }
    }
    return dp[n];
};
```

#### [剑指 Offer 63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

难度中等

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票**一次**可能获得的最大利润是多少？

**示例 1:**

```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```

代码：

```
var maxProfit = function(prices) {
    if(prices.length===0){return 0;}
    var dp = Array.from(Array(prices.length),()=>Array(2).fill(0));
    // dp[i][0]表示第i天买入时可能获得的最大利润
    // dp[i][1]表示第i天卖出时可能获得的最大利润
    dp[0][0] = -prices[0]; // 第0天买入
    dp[0][1] = 0;    // 第0天卖出
    for(let i=1;i<prices.length;i++){
        // 第i天买入    第i-1天买入   第i天卖入
        dp[i][0] = Math.max(dp[i-1][0],-prices[i]);
        // 第i天卖出    第i-1天买入   第i-1天卖出
        dp[i][1] = Math.max(dp[i-1][0]+prices[i],dp[i-1][1]);
    }
    return dp[prices.length-1][1];
};
```

优化：

```
var maxProfit = function(prices) {
    if(prices.length===0){return 0;}
    dp0 = -prices[0]; // 第0天买入
    dp1 = 0;    // 第0天卖出
    for(let i=1;i<prices.length;i++){
        // 第i天买入    第i-1天买入   第i天卖入
        dp0 = Math.max(dp0,-prices[i]);
        // 第i天卖出    第i-1天买入   第i-1天卖出
        dp1 = Math.max(dp0+prices[i],dp1);
    }
    return dp1;
};
```

#### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

难度中等

给定正整数 *n*，找到若干个完全平方数（比如 `1, 4, 9, 16, ...`）使得它们的和等于 *n*。你需要让组成和的完全平方数的个数最少。

给你一个整数 `n` ，返回和为 `n` 的完全平方数的 **最少数量** 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

**示例 1：**

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

**示例 2：**

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```

```
var numSquares = function(n) {
    // dp[i]表示和为 i 的完全平方数的 最少数量
    let dp = Array(n+1).fill(0);
    for(let i=0;i<dp.length;i++){
        dp[i] = i;
        // j*j<=i
        for(let j=1;j*j<=i;j++){
            // 将i分解为 j*j  i-j*j
            dp[i] = Math.min(dp[i],dp[i-j*j]+1);
        }
    }
    return dp[n];
};
```

## 3 dfs

DFS使用栈结构，后进先出。深度优先搜索旨在不管有多少条岔路，先一条路走到底，不成功就返回上一个路口然后就选择下一条岔路

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

#### [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/)

难度中等552

给你一个 `m x n` 的矩阵 `board` ，由若干字符 `'X'` 和 `'O'` ，找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

 **示例 1：**

![img](https://assets.leetcode.com/uploads/2021/02/19/xogrid.jpg)

想法：从每一个是O的边界元素出发，找到与该元素连接的所有同为O的元素，将其修改为#。遍历最终的board,将所有#还原为O,将所有O变为X

dfs:

1. 从某一个值为O的边界元素出发，将其修改为#
2. 找到与该边界元素相邻的值为O的第一个元素，将其修改为#，重复执行2，直至指针越界或者找到的值为#或者X
3. 找到与该边界元素相邻的值为O的第二个元素，....

总之：dfs就是一条道走到黑，再换下一条道

```
var solve = function(board){
    // 将从某个元素开始的所有相连的O元素修改为#
    var dfs = function(board,  i,  j) {
        if (i < 0 || j < 0 || i >= board.length  || j >= board[0].length || board[i][j] == 'X' || board[i][j] == '#') {
            // board[i][j] == '#' 说明已经搜索过了. 
            return;
        }
        board[i][j] = '#';
        dfs(board, i - 1, j); // 上
        dfs(board, i + 1, j); // 下
        dfs(board, i, j - 1); // 左
        dfs(board, i, j + 1); // 右
        return board;
    }

    if (board == null || board.length == 0) return;
    var m = board.length;
    var n = board[0].length;
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            // 从边缘o开始搜索,将与之相连的所有O变成#
            let  isEdge = i == 0 || j == 0 || i == m - 1 || j == n - 1;
            if (isEdge && board[i][j] == 'O') {
                dfs(board, i, j);
            }
        }
    }
    // 将修改后的board中的所有的#变成O,所有的O变成X
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            if (board[i][j] == 'O') {
                board[i][j] = 'X';
            }
            if (board[i][j] == '#') {
                board[i][j] = 'O';
            }
        }
    } 
};
```

bfs: 

1. 从某一个值为O的边界元素出发，将其修改为#
2. 记录该元素相邻(上下左右)的所有为O的元素，将对应的索引放入queue，并将其值修改为#
3. 从queue中依次取出所有的值为O的元素，重复2

总之：bfs就是一层一层处理

```
var solve = function(board){
    // 将从某个元素开始的所有相连的O元素修改为#
    var bfs = function(board,  i,  j) {
        let queue = [];
        board[i][j] = '#';
        queue.push([i,j]);
        while(queue.length>0){
            let cur = queue.shift();
            let i = cur[0], j=cur[1];
            // console.log(i,j)
            if(i-1>=0 && board[i-1][j]==='O'){
                queue.push([i-1,j]);
                board[i - 1][j] = '#';
            }
            if(i+1<=board.length-1 && board[i+1][j]==='O'){
                queue.push([i+1,j]);
                board[i + 1][j] = '#';
            }
            if(j-1>=0 && board[i][j-1]==='O'){
                queue.push([i,j-1]);
                board[i][j-1] = '#';
            }
            
            if(j+1<=board[0].length-1 && board[i][j+1]==='O'){
                queue.push([i,j+1]);
                board[i][j+1] = '#';
            }
        }
    }

    if (board == null || board.length == 0) return;
    var m = board.length;
    var n = board[0].length;
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            // 从边缘o开始搜索,将与之相连的所有O变成#
            let  isEdge = i == 0 || j == 0 || i == m - 1 || j == n - 1;
            if (isEdge && board[i][j] == 'O') {
                bfs(board, i, j);
            }
        }
    }
    // 将修改后的board中的所有的#变成O,所有的O变成X
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            if (board[i][j] == 'O') {
                board[i][j] = 'X';
            }
            if (board[i][j] == '#') {
                board[i][j] = 'O';
            }
        }
    } 
};
```

#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

难度中等

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

**示例 1：**

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**注意：如果grid中的元素不会影响下一个值，则可以直接修改当前元素，表示当前元素已经访问过了**

dfs解法：

```
var numIslands = function(grid) {
    // 将与某个元素1相连的所有1标记为2
    var dfs = function(grid,  i,  j) {
        if (i < 0 || j < 0 || i >= grid.length  || j >= grid[0].length || grid[i][j] == '0' || grid[i][j] == '2') {
            // grid[i][j] == '2' 说明已经搜索过了. 
            return;
        }
        grid[i][j] = "2";
        dfs(grid, i - 1, j); // 上
        dfs(grid, i + 1, j); // 下
        dfs(grid, i, j - 1); // 左
        dfs(grid, i, j + 1); // 右
    }

    if (grid == null || grid.length == 0) return 0;
    var m = grid.length;
    var n = grid[0].length;
    // var visited = Array.from(Array(m),()=>Array(n).fill(false));
    let index = 0;
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            // 从边缘o开始搜索,将与之相连的所有O变成#
            if (grid[i][j] == '1') {
                dfs(grid, i, j);
                index++;
            }
        }
    }
    return index;
};
```

bfs解法：

```
var numIslands = function(grid) {
    var bfs = function(grid,  i,  j) {
        let queue = [];
        queue.push([i,j]);
        grid[i][j] = 2;
        while(queue.length>0){
            let cur = queue.shift();
            let i = cur[0], j=cur[1];
            if(i-1>=0 && grid[i-1][j]==='1'){
                queue.push([i-1,j]);
                grid[i-1][j] = 2;
            }
            if(i+1<=grid.length-1 && grid[i+1][j]==='1'){
                queue.push([i+1,j]);
                grid[i+1][j] = 2;
            }
            if(j-1>=0 && grid[i][j-1]==='1'){
                queue.push([i,j-1]);
                grid[i][j-1] = 2;
            }
            
            if(j+1<=grid[0].length-1 && grid[i][j+1]==='1'){
                queue.push([i,j+1]);
                grid[i][j+1] = 2;
            }
        }
    }

    if (grid == null || grid.length == 0) return 0;
    var m = grid.length;
    var n = grid[0].length;
    let index = 0;
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            // 从边缘o开始搜索,将与之相连的所有O变成#
            if (grid[i][j] == '1') {
                bfs(grid, i, j);
                index++;
            }
        }
    }
    return index;
};
```

## BFS

BFS找到的路径一定是最短的，但是空间复杂度比DFS高

BFS使用队列结构，先进先出

广度优先搜索旨在面临一个路口时，把所有的岔路口都记下来，然后选择其中一个进入，然后将它的分路情况记录下来。然后再进入另外一个岔路，并重复这样的操作

模板：

```
BFS使用队列，把每个还没有搜索到的点依次放入队列中，然后再弹出对头的元素当做当前遍历点。
let queue = [];
let level = 0;
while queue 不空{	
	let len = queue.length;
	for(let i=0;i<len;i++){
		let cur = queue.pop();
		if(cur is target) { return level;}
		if(cur 有效并且没有被访问过){ queue.push(cur); }
	}
	level++;
}
```

#### [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

难度中等

给你一个二叉树，请你返回其按 **层序遍历** 得到的节点值。 （即逐层地，从左到右访问所有节点）。

示例：
二叉树：[3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层序遍历结果：

[
  [3],
  [9,20],
  [15,7]
]

```
function levelOrder(root){
	if(root === null){ return []; }
    let queue = [root];
    let res = [];   // 存放最终的结果
    let level = 0;   // 存放二叉树的高度
    while(queue.length){	
        let len = queue.length;
        let res1 = [];  // 存放每一层的遍历结果
        for(let i=0;i<len;i++){
            let cur = queue.shift();
            res1.push(cur);
            if(cur.left!==null){ queue.push(cur.left); }
            if(cur.right!==null){ queue.push(cur.right); }
        }
        res.push(res1);
        level++;
    }
    return res;
}
```

#### [103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

难度中等453

给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

```
var zigzagLevelOrder = function(root) {
    if(root===null){return [];}
    var queue = [root]; //保存每一层的节点
    var res = [];
    let index = 0;
    while(queue.length){
        index++;
        var arr = [];
        let len = queue.length;
        while(len){
            let node = queue.shift();
            arr.push(node.val);
            if(node.left !== null){
                queue.push(node.left);
            }
            if(node.right !== null){
                queue.push(node.right);
            }
            
            len--;
        }
        // 偶数层遍历结果取反
        if(index%2===0){
            arr.reverse();
        }
        res.push(arr);
    }
    return res;
};
```



#### [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

难度简单

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

**说明：**叶子节点是指没有子节点的节点。

```
function minDepth(root){
	if(root === null){ return 0; }
    let queue = [];
    let level = 0;   // 存放二叉树的高度
    while(queue.length){	
        let len = queue.length;
        for(let i=0;i<len;i++){
            let cur = queue.shift();
            if(res.left===null && left.right===null){
            	return level;
            }
            if(res.left!==null){ queue.push(res.left); }
            if(res.right!==null){ queue.push(res.right); }
        }
        level++;
    }
    return res;
}
```

当然这个问题也可以使用递归解决：

```
function minDepth(root){
	if(root === null){ return 0; }
    let left = minDepth(root.left);
    let right = minDepth(root.right);
    if(!left){return right+1;}
    if(!right){return left+1;}
    return Math.min(left,right);
}
```

#### [752. 打开转盘锁](https://leetcode-cn.com/problems/open-the-lock/)

难度中等

你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： `'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'` 。每个拨轮可以自由旋转：例如把 `'9'` 变为 `'0'`，`'0'` 变为 `'9'` 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 `'0000'` ，一个代表四个拨轮的数字的字符串。

列表 `deadends` 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 `target` 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。

示例 1:

```
输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。
```

BFS代码：

参考：https://mp.weixin.qq.com/s/WH_XGm1-w5882PnenymZ7g

比如说从`"0000"`开始，转一次，可以穷举出`"1000", "9000", "0100", "0900"...`共 8 种密码。然后，再以这 8 种密码作为基础，对每个密码再转一下，穷举出所有可能…

```
// 将s[i]向上拨一下
function plusOne(s,j){ 
	if(s[j]==='9'){ 
		// 将s[j]处的元素修改为0
		s1 = s.slice(0,j)+"0"+s.slice(j+1);
	}else{
		s1 = s.slice(0,j)+(Number(s[j])+1)+s.slice(j+1);
	}   
    return s1;
}

// 将s[i]向下拨一下
function minusOne(s,j){ 
	if(s[j]==='0'){ 
		// 将s[j]处的元素修改为9
		s1 = s.slice(0,j)+"9"+s.slice(j+1);
	}else{
		s1 = s.slice(0,j)+(Number(s[j])-1)+s.slice(j+1);
	}   
    return s1;
}

let openLock = function(deadends, target){
	// 使用一个set保存所有的死亡数字，方便后面查找
	let deads = new Set();
	for(let i=0;i<deadends.length;i++){
		deads.add(deadends[i]);
	}
	// 记录已经穷举过的代码
	let visited  = new Set();
	let q = ["0000"];   // 记录每拨动一次后所有可能的q的结果
	let step = 0;   // 记录拨动的次数
	visited.add("0000");  // 记录已经遍历的密码
	
	while (q.length!==0) {
        let sz = q.length;
        /* 将当前队列中的所有节点向周围扩散 */
        for (let i = 0; i < sz; i++) {
            let cur = q.shift();

            /* 判断是否到达终点 */
            if (deads.has(cur)){continue;}   // 如果是死亡数字，则执行下一个可能数值
            if (cur === target){return step;}

            /* 将一个节点的未遍历相邻节点加入队列，记录再拨一次后可能出现的结果 */
            for (let j = 0; j < 4; j++) {  // 每次拨动的位置有4个选择
                let up = plusOne(cur, j);
                if (!visited.has(up)) {
                    q.push(up);
                    visited.add(up);
                }
                let down = minusOne(cur, j);
                if (!visited.has(down)) {
                    q.push(down);
                    visited.add(down);
                }
            }
        }
        console.log(q)
        /* 在这里增加步数 */
        step++;
    }
    // 如果穷举完都没找到目标密码，那就是找不到了
    return -1;
}
```

每一层的q:

```
[
  '1000', '9000',
  '0100', '0900',
  '0010', '0090',
  '0001', '0009'
]
[
  '2000', '1100', '1900', '1010',   
  '1090', '1001', '1009',         // 1000的下一个可能数值
  '8000',
  '9100', '9900', '9010', '9090',
  '9001', '9009',                 // 0900的下一个可能数值
  '0200', '0110',
  '0190', '0101', '0109', '0800',
  '0910', '0990', '0901', '0909',
  '0020', '0011', '0019', '0080',
  '0091', '0099', '0002', '0008'
]
[
  '3000', '2100', '2900', '2010', '2090', '2001', '2009',
  '1200', '1110', '1190', '1101', '1109', '1800', '1910',
  '1990', '1901', '1909', '1020', '1011', '1019', '1080',
  '1091', '1099', '1002', '1008', '7000', '8100', '8900',
  '8010', '8090', '8001', '8009', '9200', '9110', '9190',
  '9101', '9109', '9800', '9910', '9990', '9901', '9909',
  '9020', '9011', '9019', '9080', '9091', '9099', '9002',
  '9008', '0300', '0210', '0290', '0201', '0209', '0120',
  '0111', '0119', '0180', '0191', '0199', '0108', '0700',
  '0810', '0890', '0801', '0809', '0920', '0911', '0919',
  '0980', '0991', '0999', '0902', '0908', '0030', '0021',
  '0029', '0012', '0018', '0070', '0081', '0089', '0092',
  '0098', '0102', '0003', '0007'
]
[
  '4000', '3100', '3900', '3010', '3090', '3001', '3009',
  '2200', '2110', '2190', '2101', '2109', '2800', '2910',
  '2990', '2901', '2909', '2020', '2011', '2019', '2080',
  '2091', '2099', '2002', '2008', '1300', '1210', '1290',
  '1201', '1209', '1120', '1111', '1119', '1180', '1191',
  '1199', '1102', '1108', '1700', '1810', '1890', '1801',
  '1809', '1920', '1911', '1919', '1980', '1991', '1999',
  '1902', '1908', '1030', '1021', '1029', '1012', '1018',
  '1070', '1081', '1089', '1092', '1098', '1003', '1007',
  '6000', '7100', '7900', '7010', '7090', '7001', '7009',
  '8200', '8110', '8190', '8101', '8109', '8800', '8910',
  '8990', '8901', '8909', '8020', '8011', '8019', '8080',
  '8091', '8099', '8002', '8008', '9300', '9210', '9290',
  '9201', '9209', '9120', '9111', '9119', '9180', '9191',
  '9199', '9102',
  ... 91 more items
]
[
  '5000', '4100', '4900', '4010', '4090', '4001', '4009',
  '3200', '3110', '3190', '3101', '3109', '3800', '3910',
  '3990', '3901', '3909', '3020', '3011', '3019', '3080',
  '3091', '3099', '3002', '3008', '2300', '2210', '2290',
  '2201', '2209', '2120', '2111', '2119', '2180', '2191',
  '2199', '2102', '2108', '2700', '2810', '2890', '2801',
  '2809', '2920', '2911', '2919', '2980', '2991', '2999',
  '2902', '2908', '2030', '2021', '2029', '2012', '2018',
  '2070', '2081', '2089', '2092', '2098', '2007', '1400',
  '1310', '1390', '1301', '1309', '1220', '1211', '1219',
  '1280', '1291', '1299', '1202', '1208', '1130', '1121',
  '1129', '1112', '1118', '1170', '1181', '1189', '1192',
  '1198', '1103', '1107', '1600', '1710', '1790', '1701',
  '1709', '1820', '1811', '1819', '1880', '1891', '1899',
  '1802', '1808',
  ... 256 more items
]
[
  '5100', '5900', '5010', '5090', '5001', '5009', '4200',
  '4110', '4190', '4101', '4109', '4800', '4910', '4990',
  '4901', '4909', '4020', '4011', '4019', '4080', '4091',
  '4099', '4002', '4008', '3300', '3210', '3290', '3201',
  '3209', '3120', '3111', '3119', '3180', '3191', '3199',
  '3102', '3108', '3700', '3810', '3890', '3801', '3809',
  '3920', '3911', '3919', '3980', '3991', '3999', '3902',
  '3908', '3030', '3021', '3029', '3012', '3018', '3070',
  '3081', '3089', '3092', '3098', '3003', '3007', '2400',
  '2310', '2390', '2301', '2309', '2220', '2211', '2219',
  '2280', '2291', '2299', '2202', '2208', '2130', '2121',
  '2129', '2112', '2118', '2170', '2181', '2189', '2192',
  '2198', '2103', '2107', '2600', '2710', '2790', '2701',
  '2709', '2820', '2811', '2819', '2880', '2891', '2899',
  '2802', '2808',
  ... 477 more items
]

```

#### [207. 课程表](https://leetcode-cn.com/problems/course-schedule/)

难度中等

你这个学期必须**选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1` **。

在选修某些课程之前需要一些先修课程。 先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]` ，表示如果要学习课程 `ai` 则 **必须** 先学习课程 `bi` 。

- 例如，先修课程对 `[0, 1]` 表示：想要学习课程 `0` ，你需要先完成课程 `1` 。

请你判断是否可能完成所有课程的学习？如果可以，返回 `true` ；否则，返回 `false` 。

```
const canFinish = (numCourses, prerequisites) => {
  const inDegree = new Array(numCourses).fill(0); // 入度数组
  const map = {};            // 邻接表
  // 求每门课的初始入度值                           
  for (let i = 0; i < prerequisites.length; i++) {
      // prerequisites的第一列的元素肯定都有一个入度，就是第二列产生的
    inDegree[prerequisites[i][0]]++;              
    if (map[prerequisites[i][1]]) {               // 当前课已经存在于邻接表
      map[prerequisites[i][1]].push(prerequisites[i][0]); // 添加依赖它的后续课
    } else {               
      // 当前课不存在于邻接表，则将后续的依赖课程prerequisites[i][0]放入依赖中                       
      map[prerequisites[i][1]] = [prerequisites[i][0]];
    }
  }
//   console.log(map,inDegree)
// { '0': [ 3 ], '1': [ 3, 4 ], '2': [ 4 ], '3': [ 5 ], '4': [ 5 ] } 
// [ 0, 0, 0, 2, 2, 2 ]
  const queue = [];   // [0 1 2]
  for (let i = 0; i < inDegree.length; i++) { // 所有入度为0的课入列
    if (inDegree[i] == 0) queue.push(i);
  }
  let count = 0;
  while (queue.length) {
    const selected = queue.shift();           // 当前选的课，出列
    count++;                                  // 选课数+1
    const toEnQueue = map[selected];          // 获取这门课对应的后续课
    if (toEnQueue && toEnQueue.length) {      // 确实有后续课
      // // 依次取出依赖当前课程的课程，将它们的入度减1
      for (let i = 0; i < toEnQueue.length; i++) {
        inDegree[toEnQueue[i]]--;             // 依赖它的后续课的入度-1
        if (inDegree[toEnQueue[i]] == 0) {    // 如果因此减为0，入列
          queue.push(toEnQueue[i]);
        }
      }
    }
  }
  return count == numCourses; // 选了的课等于总课数，true，否则false
};
```

![](./img/05.png)

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

#### [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

简单

输入一个**递增排序**的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

示例 1：

输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]

```
var twoSum = function(nums, target) {
    // 双指针
    let i=0,j=nums.length-1;
    while(i<j){
        if(nums[i]+nums[j]>target){
            j--;
        }else if(nums[i]+nums[j]<target){
            i++;
        }else{
            return [nums[i],nums[j]];
        }
    }
};
```

## 5 二叉树

#### [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

分析：二叉搜索树，即中序遍历为增序排列

中序遍历：左 根 右

则右 根 左就是降序排列，得到降序排列的第k个即可

```
var kthLargest = function(root, k) {
    // 方法2：堆排序
    // 方法1：中序遍历，得到第k个
    let count = 0,res=null;
    var traverse = function(root){
        if(root===null){return ;}
        traverse(root.right);
        count++;
        if(count===k){
            res = root.val;
            return ;
        }
        traverse(root.left);
    }
    traverse(root);
    return res;
};
```

#### [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

难度简单

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

```
var lowestCommonAncestor = function(root, p, q) {
    if(root===null){return null;}
    if(root === p || root===q){
        return root;
    }
    let p1 = lowestCommonAncestor(root.left,p,q);
    let p2 = lowestCommonAncestor(root.right,p,q);
    // 左右子树中均存在，则表明root是当前的最近公共节点
    if(p1 && p2){
        return root;
    }
    if(p1){return p1;}
    return p2;
};
```

#### [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

难度简单

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

```
当然直接用上面68二的方法也可以，但是考虑到是二叉搜索树，可以利用它的性质：
var lowestCommonAncestor = function(root, p, q) {
    // 先获取到两者之中的大和小
    let max = p.val > q.val ? p : q;
    let min = p.val > q.val ? q : p;
    
    let dfs = function(root,p,q){
        if(root===null){return null;}
        if(root.val<=max.val && root.val>=min.val){
            return root;
        }
        if(root.val>max.val){
            return dfs(root.left,p,q);
        }
        return dfs(root.right,p,q);
    }
    return dfs(root,min,max);
};
```

#### [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

1. 排序链表： 节点应从小到大排序，因此应使用 中序遍历 “从小到大”访问树的节点。
2. 双向链表： 在构建相邻节点的引用关系时，设前驱节点 pre 和当前节点 cur ，不仅应构建 pre.right = cur ，也应构建 cur.left = pre 。
3. 循环链表： 设链表头节点 head 和尾节点 tail ，则应构建 head.left = tail 和 tail.right = head 。

链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/solution/mian-shi-ti-36-er-cha-sou-suo-shu-yu-shuang-xian-5/

```
var treeToDoubleList = function(root){
	if(root===null){return null;}
	let pre=null,head=null;
	dfs(root);
	// 连接收尾节点
	pre.right = head;
	head.left = pre;
	var dfs = function(root){
		if(root===null){return;}
		dfs(root.left);
		if(pre===null){
			head = root;
		}else{
			pre.right = root;
		}
		root.left = pre;
		pre = root;
		dfs(root.right);
	}
}
```

## 6 位运算

这里涉及到js中的位运算符，位运算符只对整数起作用。js中所有数都是以64位浮点数的形式存储，但是做位运算时，会先将数值转化为32位带符号的整数，位运算的结果也是一个32位带符号的整数。

小数转为带符号整数：直接将小数位去除，只取整数位

将数值num转换为32位带符号整数：`num=num | 0;`,无论num是整数还是小数

第32位是符号位，所以有符号整数的范围是：-2^31 ~ 2^31-1

```
1|0    1
-1|0   -1
Math.pow(2,32)|1   0  // 2^32 是二进制位的第33位，溢出，直接被截断
(Math.pow(2,32)+1)|1   1      
(Math.pow(2,32)-1)|1   -1
```

左移运算符：`<<`,尾部补0，最高位的符号位一起移动，左移i位相当于 `num*(2^i)`

右移运算符：>>,正数头部补0，负数头部补1，最高位参与移动，左移i位相当于 `Math.floor(num/(2^i))`。

无符号右移：`>>>`,头部一律补0，不考虑符号位，此运算总是得到正值。

比较常用的数值位运算操作：

```
n&(n-1)  去除n的二进制位中最低的那一位1
n&(-n)   得到n的二进制位中最低的那一位1
& 与    &=
| 或    |=
^ 异或  ^=
! 非   
```

#### [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

简单

分析：异或：同为0，不同为1

相同的两个数各个位上均相同，则其异或的结果为1；所以将nums中所有的数均异或一遍，结果就相当于两个只出现一次的数a,b异或的结果

```
var singleNumber = function(nums) {
    return nums.reduce((a,b)=>a^b);
};
```

#### [260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/)

#### [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

中等

首先得到diff的最低位的1，diff1 = diff & -diff; 该位为1则表示：a,b在这一位的值是不同的。至于其它出现两次的num,则他们在这一位的值肯定是相同的，相异或的结果是0，所以就只会剩下a或者b

 diff1只有1位为0，其余位均为1，则通过求diff1&num即可将所有的num分为两组，每一组的结果相异或即可得到一个数，两组的结果合并即是所求

例如:1,2,3,4,2,1

diff = 1^2^3^4^2^1 = 3^4 = 0110

diff1 = 0010

1&diff1=0   2&diff1=1  3&diff1=1 4&diff1=0

所以num被分为了两组：1 4 1  和  2 3 2   两组分别异或的结果是4 和  3

```
var singleNumber = function(nums) {
    var diff = nums.reduce((a,b)=>{
        return a^b;
    })
    diff = diff & -diff;
    let res = [0,0];
    for(let i=0;i<nums.length;i++){
        if(diff&nums[i]){
            res[0] ^= nums[i];
        }else{
            res[1] ^= nums[i];
        }
    }
    return res;
};
```

#### [137. 只出现一次的数字 II](https://leetcode-cn.com/problems/single-number-ii/)

给你一个**整数数组** `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次 。**请你找出并返回那个只出现了一次的元素。

中等

js中整数都是以32位二进制位存储的，统计每一位上所有num的数值(1或者0)的和，最后除以3取余，然后每一位的和加起来即可得到最终的结果

```
var singleNumber = function(nums) {
    let res = 0;
    for(let i=0;i<32;i++){
        let count = 0;
        for(let num of nums){
            count += (num>>>i)&1;
        }
        // console.log(count)
        if(count%3){
            res |= 1<<i;
        }
    }
    return res;
};
```

#### [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

难度简单

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

题解：[面试题65. 不用加减乘除做加法（位运算，清晰图解）](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/solution/mian-shi-ti-65-bu-yong-jia-jian-cheng-chu-zuo-ji-7/)

 ```
 var add = function(a, b) {
     while(b!==0){
         // 进位
         c = (a&b)<<1;
         // 非进位和
         a ^= b;
         // 保存进位
         b = c;
     }
     return a;
 };
 ```

![](./img/02.png)

## 7 滑动窗口：

#### [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

示例 1：

输入：target = 9
输出：[[2,3,4],[4,5]]

思路：左开右闭滑动窗口，两个指针控制滑动窗口的大小，如果滑动窗口中的数值和小于target,则数值和加上nums[j],j右移；如果大于target则和减去nums[i],i左移。如果等于，则保存结果，并且数值和减去nums[i],加上nums[j],i、j分别右移一位

```
var findContinuousSequence = function(target) {
    let arr = [];
    for(let i=0;i<target;i++){arr[i]=i+1;}
    let sum = 0;
    let res = [];
    let i=0,j=0;
    while(i<arr.length/2){
        if(sum<target){
            sum+=arr[j];
            j++;
        }else if(sum>target){
            sum-=arr[i];
            i++;
        }else{
            res.push(arr.slice(i,j));
            sum += arr[j];
            sum -= arr[i];
            i++;
            j++;
        }
    }
    return res;
};
```

#### [剑指 Offer 64. 求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

难度中等337

求 `1+2+...+n` ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

思路：三种解决办法：

1. n*(n+1)/2
2. for或者while循环
3. &&
   1. A && B  A的结果为true则执行B，结果就是B的结果
   2. A || B     A的结果为true,则直接返回true         A的结果为false,则返回B 的结果

 ```
 n>1,则递归执行dfs(n-1),如果n<1,即当n=1时，直接返回res+1,依次添加别的值
 var sumNums = function(n) {
     let res = 0;
     let dfs = function(n){
         n>1 && dfs(n-1)>0;
         res += n;
         return res;
     }
     dfs(n);
     return res;
 };
 ```

#### [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

难度困难276

给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。

**示例:**

```
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

注意：**你可以假设 *k* 总是有效的，在输入数组不为空的情况下，1 ≤ k ≤ 输入数组的大小。**

暴力解法：O(nk)

```
var maxSlidingWindow = function(nums, k) {
    if(nums.length===0){
        return [];
    }
    let i=0,j=k-1,maxArr=[],max=nums[0];
    while(j<nums.length){
        // 假定新的滑动窗口中的最大值是滑动窗口的第一个数
        max = nums[i];
        // 求当前胡滑动窗口中的最大值
        let p=i;
        while(p<=j){
            if(max<nums[p]){
                max=nums[p];
            }
            p++;
        }
        // 保存当前滑动窗口中的最大值
        maxArr.push(max);
        // 去到下一个滑动窗口
        i++;
        j++; 
    }
    return maxArr;
};
```

单调栈解法：O(n)

单调栈实际就是栈，只不过利用了一些巧妙的逻辑，使得每次新元素入栈后，占你元素都保持有序。

在单次滑动窗口中获取最大值是O(1)的时间复杂度，单调队列就是递增或者递减，每次获取直接取得队头元素或者队尾元素即可

```
var maxSlidingWindow = function(nums, k) {
    let res = [];
    // 存放nums数组元素的下标，最左边下标对应的nums的值最大，dq对应的nums的元素是单调递减的
    let dq = []; 
 
    for (let i = 0; i < nums.length; i++) {
        // 当滑动窗口内的元素数量超出了k时，则删除最左侧的元素
        // 滑动窗口最右侧索引为i,则最左侧的索引最小应该为i-k+1,如果小于i-k+1,则说明滑动窗口的长度大于k了
        if (dq.length && dq[0] < i-k+1) {
            dq.shift();
        }
        // 维持单调递减
        // 新加入的元素比dq(从右往左，队尾)元素的对应的nums值大，则删除dq的元素
        while(dq.length && nums[dq[dq.length - 1]] < nums[i]) {
            dq.pop();
        }

        // 加入新元素的下标，保证dq中的索引对应的元素是递减排序的
        dq.push(i);

        // 判断是否等于或超过第一个窗口，是的话加入最大元素nums[dq[0]]
        // 当滑动窗口内的元素数量真好满足要求时，就将队头元素入栈
        if (i >= k - 1) {
            res.push(nums[dq[0]]);
        }
    }

    return res;
};

```

#### [剑指 Offer 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]

```
var MaxQueue = function() {
    this.queue = [];
    this.maxQueue = [];
};

/**
 * @return {number}
 */
MaxQueue.prototype.max_value = function() {
    // maxQueue的队头元素一定是当前queue中的最大元素
    return this.queue.length ? this.maxQueue[0] : -1;
};

/** 
 * @param {number} value
 * @return {void}
 */
MaxQueue.prototype.push_back = function(value) {
    // 去除比当前值小的最大值，保证队头元素是当前queue中最大的元素
    // 保证最大值队列是一个单调递减队列
    while(value>this.maxQueue[this.maxQueue.length-1]){
        this.maxQueue.pop();
    }
    this.queue.push(value);
    // 将当前值放入最大值队列中
    this.maxQueue.push(value);
};

/**
 * @return {number}
 */
MaxQueue.prototype.pop_front = function() {
    if (this.queue.length === 0) {
        return -1;
    }
    let temp = this.queue.shift();
    if (this.maxQueue[0] === temp) {
        this.maxQueue.shift();
    }
    return temp;
};
```

<img src='./img/04.png' height="400px" width='400px' />

## 9 map

#### [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

难度简单

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，大小王可以看成任意数字（即0可以看成是任何数字）。A 不能视为 14。

**示例 1:**

输入: [1,2,3,4,5]
输出: True

分析：分析可知数字的范围为：【0,13】

1. 没有大小王的情况：顺子一定存在max-min<=4
2. 存在大小王，但不将0考虑进去：
   1. 则顺子一定存在max-min<4
   2. 另外，排除一种情况：[1,13]中存在重复数值，则一定不能构成顺子

```
var isStraight = function(nums) {
    var map = new Map();
    let max = Number.MIN_SAFE_INTEGER,min=Number.MAX_SAFE_INTEGER;
    for(let i=0;i<nums.length;i++){
        // 遇到0，即大小王，则不算
        if(nums[i]===0){continue;}
        // 如果[1,13]中某个元素已经统计过，则必定不能构成重复元素
        if(map.has(nums[i])){return false;}
        map.set(nums[i],1);
        // 计算最大值和最小值
        max = Math.max(max,nums[i]);
        min = Math.min(min,nums[i]);
    }
    // 最大值和最大小值差值小于等于4则说明可以连成顺子
    return max-min<=4;
};
```

## 10 数组

#### [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

难度简单

0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

**示例 1：**

输入: n = 5, m = 3
输出: 3

**限制:**

- `1 <= n <= 10^5`
- `1 <= m <= 10^6`

分析：

1. 第一个删除的索引是idx=(m-1%n),因为索引从0开始。
2. 删除第一个数后，数组元素个数变成n-1,而idx处的后一位元素又会补位，所以第二位删除的元素索引就变成了(idx+m-1)%(n-1)
3. 所以被删除元素的索引就是：idx = （idx+m-1)%(n--)
4. 另外，这个题目需要注意：参数的数量级问题。以上思路的解题时间复杂度为O(n^2),则时间复杂度达到了10^10，在leetcode中使用了9260ms，接近1秒。**好像超出1s就超时了？？？**

```
var lastRemaining = function(n, m) {
    // 假定第一个删除的数字的索引是idx,则第二个删除的索引应该是(idx+m-1)%(n-1)
    let arr =Array(n);
    for(let i=0;i<n;i++){
        arr[i] = i;
    }
    let idx = 0;
    while(n>1){
        idx = (idx+m-1)%(n--);
        arr.splice(idx,1);
    }
    // console.log(arr)
    return arr[0];
};
```

#### [剑指 Offer 66. 构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

难度中等123

给定一个数组 `A[0,1,…,n-1]`，请构建一个数组 `B[0,1,…,n-1]`，其中 `B[i]` 的值是数组 `A` 中除了下标 `i` 以外的元素的积, 即 `B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]`。不能使用除法。

分析：

1. 先计算每个元素前所有元素的乘积，放入left数组中   left[i]=left[0]*...*left[i-1]
2. 计算每个元素后所有元素的乘积，放入right数组中  right[i] = right[i+1]*...*right[len-1]
3. 两个数组元素相乘即是最终的结果

 <img src="./img/03.png" width = "500" height = "200" />

```
var constructArr = function(a) {
    let len = a.length;
    if(len===0){return [];}
    let left = Array(len).fill(1),right=Array(len).fill(1);
    // 先计算每个索引处左侧所有元素的乘积  left[i]=left[0]*...*left[i-1]
    for(let i=1;i<len;i++){
        left[i] = left[i-1]*a[i-1];
    }
    // right[i] = right[i+1]*...*right[len-1]
    for(let i=len-2;i>=0;i--){
        right[i] = right[i+1]*a[i+1];
    }
    let b=[];
    // b[i]=left[i]*right[i]
    for(let i=0;i<len;i++){
        b[i] = left[i]*right[i];
    }
    return b;
};
```

## 11 字符串

#### [剑指 Offer 67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

``` 
"    123  123 abc"   ->123
"-1234  12 bac   "   ->-1234
"+123avs"            ->123
"-000123"            ->0

根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
```

代码：

```
var strToInt = function(str) {
    // 1. 过滤掉前后的空格
    str = str.replace(/^\s+/g,'');
    str = str.replace(/(\s+)$/g,'');
    let min = -1*(2**31);
    let max = 2**31-1;
    let num;
    // 2.去除头部的0
    let i=0,flag=0,flag2=1;
    while(str[i]===0){
        i++;
    }
    if(str[i]>="0" && str[i]<="9"){
        flag = i;
        while(str[i]>="0" && str[i]<="9"){
            i++;
        }
        num = Number(str.slice(flag,i));
    }else if(str[i]==='-' || str[i]==='+'){
        flag2 = str[i]==="-" ? -1 : 1;
        i++;
        // 去除头部的
        while(str[i]===0){
            i++;
        }
        flag = i;
        while(str[i]>="0" && str[i]<="9"){
            i++;
        }
        num =  i>flag ? flag2*Number(str.slice(flag,i)) : 0;
        // console.log(num)
    }else{
        num = 0;
    }
    return num>max ? max : num<min ? min : num;
}
```

## 排序算法：

一亿个数找到最大的其中1000个数：**要求效率高并且空间占用低**

思路1：使用快速排序将所有数的顺序排列好，再取出前1000个数。快速排序的思路是每次选取一个index,排序一次使得大于该索引处的元素均大于该元素，小于index索引的元素均小于该元素。所以如果是找第k个最大的元素，则直接判断index等于k-1时说明找到了第k个最大的元素。快速排序的时间复杂度O(nlogn),就是10^8log2 10^8,快速排序空间复杂度为O(1)

思路2：使用堆排序，最大堆，每次都找到最大的元素，首先将数据集合构造成堆（自下向上构造堆的时间复杂度为o(n)），将最大值first与末尾数last交换位置，然后再对[first, last - 1]重建堆999次（自顶向下重建堆的时间复杂度为o(2log2n)），所以总的时间复杂度为n + 999 × log2n = (10^ 8) + 2 × 999 × 27，约等于10^ 8，空间复杂度为o(1)，效率高且占用内存少

[最大堆排序](https://zhuanlan.zhihu.com/p/124885051)

在堆中，索引为i的元素对应的左子节点的索引为2*i+1,右子节点为2*i+2,父结点下标为(i-1)/2

长度为len的堆的第一个非叶子节点的索引为 [i/2]-1

```
var len;

// 创建最大堆
function buildMaxHeap(arr) {
  //建堆
  len = arr.length;
  // [n/2-1]表示的是最后一个非叶子节点 (本来是n/2（堆从1数起），但是这里arr索引是从0开始，所以-1)
  let index = Math.floor(len / 2) - 1;
  // 遍历每一个非叶子节点，构建最大堆
  for (var i = index; i >= 0; i--) {
    maxHeapify(arr, i);
  }
}

// 调整以nums[i]为根节点的堆为最大堆
function maxHeapify(arr, i) {
  //堆调整
  var left = 2 * i + 1,
    right = 2 * i + 2,
    largest = i; //i为该子树的根节点

  if (left < len && arr[left] > arr[largest]) {
    largest = left;
  }

  if (right < len && arr[right] > arr[largest]) {
    largest = right;
  }

  if (largest != i) {
    //即上面的if中有一个生效了
    swap(arr, i, largest); //交换三者中的较大者作为父节点
    maxHeapify(arr, largest); //交换后，原值arr[i]（往下降了）（索引保存为largest），
    //arr[i]作为根时，子节点可能比它大，因此要继续调整
  }
}

// 交换元素
function swap(arr, i, j) {
  var temp = arr[i];
  arr[i] = arr[j];
  arr[j] = temp;
}

// 最大堆排序
function heapSort(arr) {
  // 第一步：先创建一个最大堆
  buildMaxHeap(arr);
  //   从最后一个非叶子节点开始，将最大值交换当前堆中到最后一个元素处，然后将除了该最大值之外的堆重新构建为最大堆
  for (var i = arr.length - 1; i > 0; i--) {
    swap(arr, 0, i);
    len--;
    maxHeapify(arr, 0);
  }
  return arr;
}

console.log(heapSort([3, 1, 5, 2, 7, 0, 10], 4));

```

本题中直接当len=一亿-1000时，表明前100个最大元素已经找到了，所以直接输出arr.slice(一亿-1000)即可

各个算法的时间复杂度比较：

#### [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

示例 1：

输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1

使用一个备用栈，模拟元素的入栈和出栈，从pushed数组开始，将元素依次入栈。如果某个元素是poppped中的第一个元素，则表示该元素出栈了，则将该元素出栈

```
var validateStackSequences = function(pushed, popped) {
    let stack = [],m=0;
    for(let i=0;i<pushed.length;i++){
        stack.push(pushed[i]);
        while(stack.length>0 && stack[stack.length-1]===popped[m]){
            stack.pop();
            m = m+1;
        }
    }
    return stack.length===0;
};
```

#### [剑指 Offer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

判断一个字符串中的元素是否是纯数字

```
function isOnlyNum(s){
	for(let c of s){
		if(c<"0" && c>'9'){ return false; }
	}
	return true;
}
```

判断一个字符串中的元素是否是合法的小数      .12    12.    12.12

```
function isLegalDecimal(s){
	if(s.indexOf('.')!==-1 && s.length>1){
		if(s.indexOf(".") ！== s.lastIndexOf(".")){return false;}
		if(s.charAt(0) === '.') { return isOnlyNum(s.substring(1)); }  // .12
		if(s.charAt(s.length-1) === '.') { return isOnlyNum(s.substring(1)); }  // 12.
		let index = s.indexOf(".");
		return s.length>2 && isOnlyNum(s.substring(0,index)) && isOnlyNum(s.substring(index+1));
	}
	return false;
}
```

判断是否是整数

```
function checkNum( s) {    //判断是否是整数
    if(s == null || s.length == 0) return false;
    if(s.charAt(0) == '-' || s.charAt(0) == '+') return s.length > 1 ? isOnlyNum(s.substring(1)) : false;
    return isOnlyNum(s);
}
```

判断是否是整数或者是小数

```
function checkNumOrSnum( s) {  //判断是否整数或者小数
    if(s == null || s.length == 0) return false;  // s为空
    if(checkNum(s)) return true;   // s是整数
    // s是带符号小数组成的字符串
    if(s.charAt(0) == '-' || s.charAt(0) == '+'){    
        return s.length > 1 ? isNumandPolet(s.substring(1)) : false;
    }
    // s是无符号小数  字符串
    return isNumandPolet(s);
}
```

请实现一个函数用来判断字符串是否表示**数值**（包括整数和小数）。

```
const isNumber = function( s) {
    // 先去除空格
    s = s.trim();
    /** 通过判断有没有e，有e则前面为整数或者小数，后面为整数。没有e则为整数或者小数*/
    // 有e并且只有一个e
    if(s.indexOf("e")!==-1 && s.indexOf("e") == s.lastIndexOf("e")) {
        let ss = s.split("e");   // 按照e将字符串分段
        // 如果分段的长度小于等于1，则说明不是有效数值   1e e1 e
        // 如果分段长度，
        return ss.length > 1 ? checkNumOrSnum(ss[0]) && checkNum(ss[1]) : false;
    }
    // 有E并且只有一个E
    if(s.indexOf("E")!==-1 && s.indexOf("E") == s.lastIndexOf("E")) {
        let ss = s.split("E");
        return ss.length > 1 ? checkNumOrSnum(ss[0]) && checkNum(ss[1]) : false;
    }
    // 没有e或者E  或者有多个e
    return checkNumOrSnum(s);
}
```

#### [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

示例 1:

输入: [7,5,6,4]
输出: 5

归并排序解法：

```
var reversePairs = function(nums) {
    // 归并排序
    let res = 0;
    function mergeSort(nums){
        if(nums.length<=1){
            return nums;
        }
        let len = Math.floor(nums.length/2);
        return merge(mergeSort(nums.slice(0,len)),mergeSort(nums.slice(len)));
    }
    function merge(left, right) {
        let result = [];
        let leftLen = left.length;
        let rightLen = right.length;
        let len = leftLen + rightLen;
        for(let index = 0, i = 0, j = 0; index < len; index ++) {
            if(i >= leftLen) result[index] = right[j ++];
            else if (j >= rightLen) result[index] = left[i ++];
            else if (left[i] <= right[j]) result[index] = left[i ++];
            else {
                result[index] = right[j ++];
                res += leftLen - i;//在归并排序中唯一加的一行代码
            }
        }
        return result;
    }
    mergeSort(nums);
    return res;
}
```

插入排序解法：

```
var reversePairs = function(nums) {
    // 插入排序，记录交换次数
    let res = 0;
    for(let i=1;i<nums.length;i++){
        let key = nums[i];
        let j = i-1;
        while(j>=0 && nums[j]>key){
            nums[j+1] = nums[j]
            j--;
            res++;
        }
        nums[j+1] = key;
    }
    return res;
}
```

暴力解法会超时：

```
var reversePairs = function(nums) {
// 暴力解法，会超时
    let res=0;
    for(let i=0;i<nums.length-1;i++){
        for(let j=i+1;j<nums.length;j++){
            if(nums[i]>nums[j]){res++;}
        }
    }
    return res;
};
```

#### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

难度中等

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

**中心扩散法**

```
var longestPalindrome = function(s) {
  // s为空字符串或为长为1的字符串，返回字符串本身
  if (s.length < 2) return s;

  let res = '';
  // 遍历每个可能的中心点位，以左右指针模拟中心点
  for (let i = 0; i < s.length; i++) {
    // 单数情况
    getCenter(i, i);
    // 双数情况
    getCenter(i, i + 1);
  }

  // 本函数的作用为：获取最长的，以本中心点为中心的回文串
  function getCenter(left, right) {
    // 边界条件：左指针不小于0，右指针不超过数组的最长长度。
    // 进入循环条件：满足边界条件，且当前两个指针指向的字符相等
    while (left >= 0 && right < s.length && s[left] == s[right]) {
      // 左侧指针左移，右侧指针右移，开启下次字符相等的判断循环。当超出系统边界或两指针指向的字符不相等，则退出
      left--;
      right++;
    }

    // 循环结束，两指针目前指向的字符串中间其实是不满足回文串
    // 事实上本次while获得的回文串的左侧为left + 1，右侧为right - 1
    // 所以本次获得的回文串长度为 (right - 1) - (left + 1) + 1 = right - left - 1，与res长度判断后取最长的回文子串
    if (right - left - 1 > res.length) {
      // 记住这里需要截取的是正确的回文子串，所以要消除while循环中，最后一次不满足条件的left、right的影响
      /**
       * left => left + 1
       * right - 1 => right - 1 + 1 = right
       **/
      res = s.slice(left + 1, right);
    }
  }
  return res
};
```

或者暴力解法：双重循环，判断每一段中是否是回文字符串

