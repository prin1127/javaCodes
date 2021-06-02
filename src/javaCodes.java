import java.util.*;

public class javaCodes {
    /*1.
    在一个长度为 n 的数组里的所有数字都在 0 到 n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字是重复的，
    也不知道每个数字重复几次。请找出数组中任意一个重复的数字。
    时间复杂度 O(N)，空间复杂度 O(1)
    将值为 i 的元素调整到第 i 个位置上进行求解
    public static int duplicate(int[] nums, int length) {
        if (nums == null || length <= 0)
            return -1;
        for (int i = 0; i < length; i++) {
            if (nums[i] != i) {
                if (nums[i] == nums[nums[i]]) {
                    return nums[i];
                }
                swap(nums, i, nums[i]);
            }
        }
        return -1;
    }
    private static void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }
    */

    /*2.
    给定一个二维数组，其每一行从左到右递增排序，从上到下也是递增排序。给定一个数，判断这个数是否在该二维数组中。
    要求时间复杂度 O(M + N)，空间复杂度 O(1)。
    该二维数组中的一个数，它左边的数都比它小，下边的数都比它大。因此，从右上角开始查找，
    就可以根据 target 和当前元素的大小关系来缩小查找区间，当前元素的查找区间为左下角的所有元素。

    public static boolean Find(int target, int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return false;
        int rows = matrix.length, cols = matrix[0].length;
        int r = 0, c = cols - 1; // 从右上角开始
        while (r <= rows - 1 && c >= 0) {
            if (target == matrix[r][c])
                return true;
            else if (target > matrix[r][c])
                r++;
            else
                c--;
        }
        return false;
    }
    */

    /*3.
    将一个字符串中的空格替换成 "%20"。
    在字符串尾部填充任意字符，使得字符串的长度等于替换之后的长度。因为一个空格要替换成三个字符（%20），
    因此当遍历到一个空格时，需要在尾部填充两个任意字符。
    令 P1 指向字符串原来的末尾位置，P2 指向字符串现在的末尾位置。P1 和 P2 从后向前遍历，当 P1 遍历到一个空格时，
    就需要令 P2 指向的位置依次填充 02%（注意是逆序的），否则就填充上 P1 指向字符的值。
    从后向前遍是为了在改变 P2 所指向的内容时，不会影响到 P1 遍历原来字符串的内容。

    public static String replaceSpace(StringBuffer str) {
        //直接调用函数：
        //return str.toString().replace(" ", "%20");
        int P1 = str.length() - 1;
        for (int i = 0; i <= P1; i++)
            if (str.charAt(i) == ' ')
                str.append("  ");
        int P2 = str.length() - 1;
        while (P1 >= 0 && P2 > P1) {
            char c = str.charAt(P1--);
            if (c == ' ') {
                str.setCharAt(P2--, '0');
                str.setCharAt(P2--, '2');
                str.setCharAt(P2--, '%');
            } else {
                str.setCharAt(P2--, c);
            }
        }
        return str.toString();
    }
     */

    /*4.
    输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
    public static class ListNode{
        public int val;
        public ListNode next;
        public ListNode(){}
        public ListNode(int val){
            this.val = val;
        }
    }
    //递归,时间复杂度 O(N)，空间复杂度 O(N)
    public static ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> ret = new ArrayList<>();
        if (listNode != null) {
            ret.addAll(printListFromTailToHead(listNode.next));
            ret.add(listNode.val);
        }
        return ret;
    }
    //头插法,利用链表头插法为逆序的特点
    //头结点和第一个节点的区别：头结点是在头插法中使用的一个额外节点，这个节点不存储值；第一个节点就是链表的第一个真正存储值的节点
    public static ArrayList<Integer> printListFromTailToHead2(ListNode listNode) {
        //相当于把原链表用头插法生成新链表
        ListNode head = new ListNode(-1);
        while (listNode != null) {
            ListNode memo = listNode.next;
            listNode.next = head.next;
            head.next = listNode;
            listNode = memo;
        }
        ArrayList<Integer> ret = new ArrayList<>();
        head = head.next;
        while (head != null) {
            ret.add(head.val);
            head = head.next;
        }
        return ret;
    }
    //栈，先进后出
    public static ArrayList<Integer> printListFromTailToHead3(ListNode listNode) {
        Stack<Integer> stack = new Stack<>();
        while (listNode != null) {
            stack.add(listNode.val);
            listNode = listNode.next;
        }
        ArrayList<Integer> ret = new ArrayList<>();
        while (!stack.isEmpty())
            ret.add(stack.pop());
        return ret;
    }
    */

    /*5.
    根据二叉树的前序遍历和中序遍历的结果，重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
    前序遍历的第一个值为根节点的值，使用这个值将中序遍历结果分成两部分，左部分为树的左子树中序遍历结果，右部分为树的右子树中序遍历的结果。
    时间复杂度 O(N)，空间复杂度 O(N)
    public static class TreeNode {
        public TreeNode left;
        public TreeNode right;
        public TreeNode root;
        public int val;
        public TreeNode() {
        }

        public TreeNode(int data) {
            this(null, null, data);
        }

        public TreeNode(TreeNode left, TreeNode right, int val) {
            this.left = left;
            this.right = right;
            this.val = val;
        }
    }
    public static TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length == 0 || in.length == 0) {
            return null;
        }
        TreeNode root = new TreeNode(pre[0]);
        for (int i = 0; i < in.length; i++) {
            if (in[i] == pre[0]) {
                //copyOfRange 函数，左闭右开
                root.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i + 1), Arrays.copyOfRange(in, 0, i));
                root.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i + 1, pre.length), Arrays.copyOfRange(in, i + 1, in.length));
                break;//无需再往后找，提前结束循环
            }
        }
        return root;
    }
    public static void PrintFromTopToBottom(TreeNode root) {
        //使用ArrayList实现一个队列，每次访问把一个结点入队列，并判断是否有左右子结点，如果有，也放入到队列中，每次取出这个队列的第一个结点。
        ArrayList<TreeNode> node = new ArrayList<TreeNode>();
        if(root == null){
            return ;
        }
        node.add(root);
        while(node.size() > 0){
            TreeNode treeNode = node.remove(0);
            if(treeNode.left != null){
                node.add(treeNode.left);
            }
            if(treeNode.right != null){
                node.add(treeNode.right);
            }
            System.out.println(treeNode.val);
        }
    }
    */

    /*6.
    给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。
    注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
    ① 如果一个节点的右子树不为空，那么该节点的下一个节点是右子树的最左节点；
    ② 否则，向上找第一个左链接指向的树包含该节点的祖先节点。
    public class TreeLinkNode {
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode next = null;
        TreeLinkNode(int val) {
            this.val = val;
        }
    }
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode.right != null) {
            TreeLinkNode node = pNode.right;
            while (node.left != null)//最左，所以要循环
                node = node.left;
            return node;
        } else {
            while (pNode.next != null) {
                TreeLinkNode parent = pNode.next;
                if (parent.left == pNode)//左孩子的父结点
                    return parent;
                pNode = pNode.next;//右孩子的祖先
            }
        }
        return null;
    }
    */

    /*7.
    用两个栈来实现一个队列，完成队列的 Push 和 Pop 操作。
    in 栈用来处理入栈（push）操作，out 栈用来处理出栈（pop）操作。
    一个元素进入 in 栈之后，出栈的顺序被反转。当元素要出栈时，需要先进入 out 栈，此时元素出栈顺序再一次被反转，
    因此出栈顺序就和最开始入栈顺序是相同的，先进入的元素先退出，这就是队列的顺序。
    Stack<Integer> in = new Stack<Integer>();
    Stack<Integer> out = new Stack<Integer>();
    public void push(int node) {
        in.push(node);
    }
    public int pop() throws Exception {
        if (out.isEmpty())
            while (!in.isEmpty())
                out.push(in.pop());
        if (out.isEmpty())
            throw new Exception("queue is empty");
        return out.pop();
    } */

    /*8.
    求斐波那契数列的第 n 项，n <= 39.
    递归：时间复杂度 O(2^N)
    public int Fibonacci(int n) {
        if (n==0 || n==1)
            return n;
        return Fibonacci(n-1) + Fibonacci(n-2);
    }
    动态规划：
    递归是将一个问题划分成多个子问题求解，动态规划也是如此，但是动态规划会把子问题的解缓存起来，从而避免重复求解子问题。
    时间复杂度 O(N)，空间复杂度 O(1)
    public int Fibonacci(int n) {
        if (n <= 1)
            return n;
        int pre2 = 0, pre1 = 1;
        int fib = 0;
        for (int i = 2; i <= n; i++) {
            fib = pre2 + pre1;
            pre2 = pre1;
            pre1 = fib;
        }
        return fib;
    }
     */

    /*9.
    一只青蛙一次可以跳上 1 级台阶，也可以跳上 2 级。求该青蛙跳上一个 n 级的台阶总共有多少种跳法.
    逆向思维，假设f[i]表示在第i个台阶上可能的方法数。如果从第n个台阶进行下台阶，下一步有2种可能，走到第n-1个台阶，或者走到第n-2个台阶。
    所以f[n] = f[n-1] + f[n-2]。f[0] = f[1] = 1。
    动态规划：时间复杂度 O(N)，空间复杂度 O(1)
    public int JumpFloor(int n) {
        if (n <= 2)
            return n;
        int pre2 = 1, pre1 = 2;
        int result = 1;
        for (int i = 2; i < n; i++) {
            result = pre2 + pre1;
            pre2 = pre1;
            pre1 = result;
        }
        return result;
    }
    */

    /*10.
    可以用 2*1 的小矩形横着或者竖着去覆盖更大的矩形。请问用 n 个 2*1 的小矩形无重叠地覆盖一个 2*n 的大矩形，总共有多少种方法？
    f[n] = f[n-1] + f[n-2]，初始条件f[1] = 1, f[2] =2
    动态规划：
    public int RectCover(int n) {
        if (n <= 2)
            return n;
        int pre2 = 1, pre1 = 2;
        int result = 0;
        for (int i = 3; i <= n; i++) {
            result = pre2 + pre1;
            pre2 = pre1;
            pre1 = result;
        }
        return result;
    }
     */

    /*11.
    一只青蛙一次可以跳上 1 级台阶，也可以跳上 2 级... 它也可以跳上 n 级。求该青蛙跳上一个 n 级的台阶总共有多少种跳法
    跳上 n-1 级台阶，可以从 n-2 级跳 1 级上去，也可以从 n-3 级跳 2 级上去...，那么f(n-1) = f(n-2) + f(n-3) + ... + f(0)
    同样，跳上 n 级台阶，可以从 n-1 级跳 1 级上去，也可以从 n-2 级跳 2 级上去... ，那么f(n) = f(n-1) + f(n-2) + ... + f(0)
    综上可得f(n) - f(n-1) = f(n-1)，即f(n) = 2*f(n-1)，等比数列
    public int JumpFloorII(int target) {
        //直接根据等比数列计算结果：
        //return (int) Math.pow(2, target - 1);
        int[] dp = new int[target];
        Arrays.fill(dp, 1);
        for (int i = 1; i < target; i++)//f(n) = f(n-1) + f(n-2) + ... + f(0)
            for (int j = 0; j < i; j++)
                dp[i] += dp[j];
        return dp[target - 1];
    }
    */

    /*12.
    股票的最大利润。可以有一次买入和一次卖出，那么买入必须在前。求最大收益。
    贪心算法：假设第 i 轮进行卖出操作，买入操作价格应该在 i 之前并且价格最低。
    public static int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0)
            return 0;
        int soFarMin = prices[0];
        int maxProfit = 0;
        for (int i = 1; i < prices.length; i++) {
            soFarMin = Math.min(soFarMin, prices[i]);
            maxProfit = Math.max(maxProfit, prices[i] - soFarMin);
        }
        return maxProfit;
    }*/

    /*13.
    旋转数组的最小数。把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
    输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
    在一个有序数组中查找一个元素可以用二分查找，二分查找也称为折半查找，每次都能将查找区间减半，这种折半特性的算法时间复杂度都为 O(logN)。
    当 nums[m] <= nums[h] 的情况下，说明解在 [l, m] 之间，此时令 h = m；否则解在 [m + 1, h] 之间，令 l = m + 1。

    public int minNumberInRotateArray(int[] nums) {
        if (nums.length == 0)
            return 0;
        int l = 0, h = nums.length - 1;
        while (l < h) {
            int m = l + (h - l) / 2;
            if (nums[l] == nums[m] && nums[m] == nums[h])
                return minNumber(nums, l, h);//相等元素顺序查找，eg：{1,1,1,0,1}
            else if (nums[m] <= nums[h])
                h = m;
            else
                l = m + 1;
        }
        return nums[l];
    }
    private int minNumber(int[] nums, int l, int h) {
        for (int i = l; i < h; i++)
            if (nums[i] > nums[i + 1])
                return nums[i + 1];
        return nums[l];
    }
    */

    /*16.
    整数拆分/剪绳子
    把一根绳子剪成多段，并且使得每段的长度乘积最大。
    给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。
    递归：f(n) = max(f(i)*f(n-i))
    贪心算法：根据（周长、面积...）数学推理可知尽可能裁剪长度为3的段时乘积最大，其次是长度为2的段。
    尽可能多剪长度为 3 的绳子，并且不允许有长度为 1 的绳子出现。
    如果出现了，就从已经切好长度为 3 的绳子中拿出一段与长度为 1 的绳子重新组合，把它们切成两段长度为 2 的绳子。
    public int integerBreak(int n) {
        if (n < 2)
            return 0;
        if (n == 2)
            return 1;
        if (n == 3)
            return 2;
        int timesOf3 = n / 3;
        if (n - timesOf3 * 3 == 1)//出现长度为1的段
            timesOf3--;
        int timesOf2 = (n - timesOf3 * 3) / 2;
        return (int) (Math.pow(3, timesOf3)) * (int) (Math.pow(2, timesOf2));
    }
    动态规划：
    动态规划求解问题的四个特征：
    ①求一个问题的最优解；
    ②整体的问题的最优解是依赖于各个子问题的最优解；
    ③小问题之间还有相互重叠的更小的子问题；
    ④从上往下分析问题，从下往上求解问题
    一般，动态规划有以下几种分类：
    ①最值型动态规划，比如求最大，最小值是多少
    ②计数型动态规划，比如换硬币，有多少种换法
    ③坐标型动态规划，比如在m*n矩阵求最值型，计数型，一般是二维矩阵
    ④区间型动态规划，比如在区间中求最值
    public int integerBreak(int n) {
        if (n < 2)
            return 0;
        if(n==2)
            return 1;
        if(n==3)
            return 2;

        int[] dp = new int[n+1];
        dp[1]=1;
        dp[2]=2;
        dp[3]=3;
        for (int i = 4; i <= n; i++) {
            int res=0;//记录最大的
            for (int j = 1; j <=i/2 ; j++) {
                res=Math.max(res,dp[j]*dp[i-j]);
            }
            dp[i]=res;
        }
        return dp[n];
    }
    */

    /*17.
    二进制中 1 的个数
    输入一个整数，输出该数32位二进制表示中1的个数。其中负数用补码表示。
    n&(n-1)，对从右向左的第一位1直接判断，遇到0直接略过。
    n:1101000, n-1: 1100111 那么n&(n-1): 1100000
    时间复杂度：O(M)，其中 M 表示 1 的个数。
    public int NumberOf1(int n) {
        int cnt = 0;
        while (n != 0) {
            cnt++;
            n &= (n - 1);
        }
        return cnt;
    }
     */

    /*18.
    数值的整数次方
    给定一个 double 类型的浮点数 base 和 int 类型的整数 exponent，求 base 的 exponent 次方。
    递归：将base的exponent 次方拆分为base*base的exponent/2 次方
    每次递归，exponent减半，则时间复杂度：log(N)

    public double Power(double base, int exponent) {
        if (exponent == 0)
            return 1;
        if (exponent == 1)
            return base;
        boolean isNegative = false;
        if (exponent < 0) {//exponent为负数
            exponent = -exponent;
            isNegative = true;
        }
        double pow = Power(base * base, exponent / 2);
        if (exponent % 2 != 0)//exponent为奇数
            pow = pow * base;
        return isNegative ? 1 / pow : pow;
    }
     */

    /*21.
     在 O(1) 时间内删除链表节点
    public class ListNode{
        public int val;
        public ListNode next;
        public ListNode(){}
        public ListNode(int val){
            this.val = val;
        }
    }
    public ListNode deleteNode(ListNode head, ListNode tobeDelete) {
        if (head == null || tobeDelete == null)
            return null;
        if (tobeDelete.next != null) {
            // 要删除的节点不是尾节点
            ListNode tmp = tobeDelete.next;
            tobeDelete.val = tmp.val;
            tobeDelete.next = tmp.next;
        } else {
            if (head == tobeDelete)
                // 只有一个节点
                head = null;
            else {
                ListNode cur = head;
                while (cur.next != tobeDelete)
                    cur = cur.next;//找到尾节点的前一个节点
                cur.next = null;
            }
        }
        return head;
    }
    */

    /*22.
    删除链表中重复的结点
    1->2->2->3->3->4
    1->4

    public static class ListNode{
        public int val;
        public ListNode next;
        public ListNode(){}
        public ListNode(int val){
            this.val = val;
        }
    }
    public static ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null)//空链表或仅有一个节点的链表
            return pHead;
        ListNode tmp = pHead.next;
        if (pHead.val == tmp.val) {// 当前节点是重复节点
            while (tmp != null && pHead.val == tmp.val)// 跳过值与当前节点相同的全部节点，找到第一个与当前节点不同的节点
                tmp = tmp.next;
            return deleteDuplication(tmp);// 从第一个与当前结点不同的结点继续递归
        } else {
            pHead.next = deleteDuplication(pHead.next);// 保留当前节点，从下一个节点继续递归
            return pHead;
        }
    }
    */

    /*23.
    表示数值的字符串
    请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
    例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
    分类讨论各情况：
    ①+-号后面必定为数字或后面为.（-.123 = -0.123）
    ②+-号只出现在第一位或在eE的后一位
    ③.后面必定为数字或为最后一位（233. = 233.0）
    ④eE后面必定为数字或+-号
    public boolean isNumeric(char[] str) {
        boolean point = false, exp = false; // 标志小数点和指数
        for (int i = 0; i < str.length; i++) {
            if (str[i] == '+' || str[i] == '-') {
                if (i + 1 == str.length || !(str[i + 1] >= '0' && str[i + 1] <= '9' || str[i + 1] == '.'))  // +-号后面必定为数字 或 后面为.（-.123 = -0.123）
                    return false;
                if (!(i == 0 || str[i-1] == 'e' || str[i-1] == 'E'))  // +-号只出现在第一位或eE的后一位
                    return false;
            } else if (str[i] == '.') {
                if (point || exp || !(i + 1 < str.length && str[i + 1] >= '0' && str[i + 1] <= '9'))  // .后面必定为数字 或为最后一位（233. = 233.0）
                    return false;
                point = true;
            } else if (str[i] == 'e' || str[i] == 'E') {
                if (exp || i + 1 == str.length || !(str[i + 1] >= '0' && str[i + 1] <= '9' || str[i + 1] == '+' || str[i + 1] == '-'))  // eE后面必定为数字或+-号
                    return false;
                exp = true;
            } else if (str[i] >= '0' && str[i] <= '9') {//数字
            } else {//字母、其他符号等
                return false;
            }
        }
        return true;
    }
    正则表达式[+-]?\\d*(\\.\\d+)?([eE][+-]?\\d+)?
    [] ： 字符集合
    () ： 分组
    ? ： 重复 0 ~ 1次
    + ： 重复 1 ~ n次
    * ： 重复 0 ~ n次
    . ： 任意字符
    \\. ： 转义后的 .
    \\d ： 数字
    public static boolean isNumeric(char[] str) {
        if (str == null || str.length == 0)
            return false;
        return new String(str).matches("[+-]?\\d*(\\.\\d+)?([eE][+-]?\\d+)?");
    }
//    其实这样是有问题的，即223.会输出false
//    但改为"[+-]?\\d*(\\.\\d*)?([eE][+-]?\\d+)?"，那么223.e3，会输出true
    */

    /*24.
    调整数组顺序使奇数位于偶数前面
    1,2,3,4,5->1,3,5,2,4
    需要保证奇数和奇数，偶数和偶数之间的相对位置不变。
    public void reOrderArray(int[] nums) {
        int oddCnt = 0;// 奇数个数
        for (int val : nums)
            if (val % 2 == 1)
                oddCnt++;
        int[] copy = nums.clone();
        int i = 0, j = oddCnt;
        for (int num : copy) {
            if (num % 2 == 1)
                nums[i++] = num;//从0开始放奇数个
            else
                nums[j++] = num;//从oddCnt位置开始放偶数个
        }
    }
    */

    /*25.
    链表中倒数第k个节点
    输入一个链表，输出该链表中倒数第k个结点。
    设链表的长度为 N。设两个指针 P1 和 P2，先让 P1 移动 K 个节点，则还有 N - K 个节点可以移动。
    此时让 P1 和 P2 同时移动，可以知道当 P1 移动到链表结尾时，P2 移动到 N - K 个节点处，该位置就是倒数第 K 个节点。
    public class ListNode {
        int val;
        ListNode next = null;
        ListNode(int val) {
            this.val = val;
        }
    }
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null)
            return null;
        ListNode P1 = head;
        while (P1 != null && k-- > 0)
            P1 = P1.next;//先让 P1 移动 K 个节点
        if (k > 0)//超过链表长度
            return null;
        ListNode P2 = head;
        while (P1 != null) {//让 P1 移动到链表结尾
            P1 = P1.next;
            P2 = P2.next;
        }
        return P2;//P2 移动到 N - K 个节点处
    }
     */

    /*26.
     链表中环的入口结点
     一个链表中包含环，请找出该链表的环的入口结点。要求不能使用额外的空间。
     （若允许使用额外的空间，则单链表遍历，hashset...）
    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null || pHead.next == null)
            return null;
        ListNode slow = pHead, fast = pHead;
        do {
            fast = fast.next.next;//指针 fast 每次移动两个节点
            slow = slow.next;//指针 slow 每次移动一个节点
        } while (slow != fast);//第一次相遇时停止移动
        fast = pHead;// fast 重新从头开始移动
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }//fast 和 slow 将在环入口点相遇
        return slow;
    }
    */

    /*27.
    反转链表
    迭代
    public ListNode ReverseList(ListNode head) {
        ListNode newList = new ListNode(-1);
        while (head != null) {//每次从原链表头取一个节点，加到新链表（-1）节点后面，形成倒序
            ListNode tmp = head.next;
            head.next = newList.next;
            newList.next = head;
            head = tmp;
        }
        return newList.next;//newList为（-1）节点
    }
    递归
    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode tmp = head.next;
        head.next = null;//先顺序把节点断开
        ListNode newHead = ReverseList(tmp);
        tmp.next = head;//再逆序把节点链接
        return newHead;
    }
     */

    /*28.
    合并两个排序的链表。
    输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
    递归
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null)
            return list2;
        if (list2 == null)
            return list1;
        if (list1.val <= list2.val) {
            list1.next = Merge(list1.next, list2);
            return list1;
        } else {
            list2.next = Merge(list1, list2.next);
            return list2;
        }
    }
    迭代
    public ListNode Merge(ListNode list1, ListNode list2) {
        ListNode head = new ListNode(-1);
        ListNode cur = head;
        while (list1 != null && list2 != null) {//不断把较小的值加到head节点链表里
            if (list1.val <= list2.val) {
                cur.next = list1;
                list1 = list1.next;
            } else {
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }
        if (list1 != null)//list2 == null
            cur.next = list1;
        if (list2 != null)//list1 == null
            cur.next = list2;
        return head.next;
    }
     */

    /*29.
    树的子结构
    输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
    第一步，在A中找到与B根节点值相同的节点；第二步判断以该节点为根节点的树是否包括B。
    其中第一步需要遍历树A，找到与B根节点值相同的节点，这个查找过程就是一个递归过程。
    第二步要判断以该节点为根节点的树是否包括B，这也是一个递归过程。
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
        public TreeNode(int val) {
            this.val = val;
        }
    }
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null)
            return false;
        return isSubtreeWithRoot(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }
    private boolean isSubtreeWithRoot(TreeNode root1, TreeNode root2) {
        if (root2 == null)//B结束，即在A中找到B子结构
            return true;
        if (root1 == null)//A结束
            return false;
        if (root1.val != root2.val)//根节点不相等
            return false;
        return isSubtreeWithRoot(root1.left, root2.left) && isSubtreeWithRoot(root1.right, root2.right);//继续向子树遍历
    }
    */

    /*30.
    二叉树的镜像
    public void Mirror(TreeNode root) {
        if (root == null)
            return;
        swap(root);
        Mirror(root.left);
        Mirror(root.right);
    }
    private void swap(TreeNode root) {
        TreeNode t = root.left;
        root.left = root.right;
        root.right = t;
    }
     */
    /*31.
    对称的二叉树
    请实现一个函数，用来判断一棵二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
    boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null)
            return true;
        return isSymmetrical(pRoot.left, pRoot.right);
    }
    boolean isSymmetrical(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null)
            return true;
        if (t1 == null || t2 == null)
            return false;
        if (t1.val != t2.val)
            return false;
        return isSymmetrical(t1.left, t2.right) && isSymmetrical(t1.right, t2.left);
    }
     */
    /*32.
    顺时针打印矩阵
    输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
    例如，如果输入如下4 X 4矩阵：
      1  2  3  4
      5  6  7  8
      9 10 11 12
     13 14 15 16
    则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
//    简单来说，就是不断地收缩矩阵的边界
//    定义四个变量代表范围，up、down、left、right
//    向右走存入整行的值，当存入后，该行再也不会被遍历，代表上边界的 up 加一，同时判断是否和代表下边界的 down 交错
//    向下走存入整列的值，当存入后，该列再也不会被遍历，代表右边界的 right 减一，同时判断是否和代表左边界的 left 交错
//    向左走存入整行的值，当存入后，该行再也不会被遍历，代表下边界的 down 减一，同时判断是否和代表上边界的 up 交错
//    向上走存入整列的值，当存入后，该列再也不会被遍历，代表左边界的 left 加一，同时判断是否和代表右边界的 right 交错
    public ArrayList<Integer> printMatrix(int [][] matrix) {
        ArrayList<Integer> list = new ArrayList<>();
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0){
            return list;
        }
        int up = 0;
        int down = matrix.length-1;
        int left = 0;
        int right = matrix[0].length-1;
        while(true){
            for(int col=left;col<=right;col++){// 最上面一行
                list.add(matrix[up][col]);
            }
            up++;// 向下逼近
            if(up > down) {// 判断是否越界
                break;
            }
            for(int row=up;row<=down;row++){// 最右边一列
                list.add(matrix[row][right]);
            }
            right--;// 向左逼近
            if(left > right){// 判断是否越界
                break;
            }
            for(int col=right;col>=left;col--){// 最下面一行
                list.add(matrix[down][col]);
            }
            down--;// 向上逼近
            if(up > down){// 判断是否越界
                break;
            }
            for(int row=down;row>=up;row--){// 最左边一列
                list.add(matrix[row][left]);
            }
            left++;// 向右逼近
            if(left > right){// 判断是否越界
                break;
            }
        }
        return list;
    }
     */

    /*33.
    包含min函数的栈
    定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的 min 函数（要求时间复杂度为O(1)）。
    private Stack<Integer> dataStack = new Stack<>();
    private Stack<Integer> minStack = new Stack<>();
    public void push(int node) {
        dataStack.push(node);
        minStack.push(minStack.isEmpty() ? node : Math.min(minStack.peek(), node));
    }
    public void pop() {
        dataStack.pop();
        minStack.pop();
    }
    public int top() {
        return dataStack.peek();
    }
    public int min() {
        return minStack.peek();
    }
     */

    /*34.
    栈的压入、弹出序列
    输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
    假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，
    但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
    新建一个栈，将数组A压入栈中，当栈顶元素等于数组B时，就将其出栈。当循环结束时，判断栈是否为空，若为空则返回true。
    public boolean IsPopOrder(int[] pushSequence, int[] popSequence) {
        if(pushSequence.length==0 || popSequence.length==0 || popSequence.length!=pushSequence.length)
            return false;
        int n = pushSequence.length;
        Stack<Integer> stack = new Stack<>();
        for (int pushIndex = 0, popIndex = 0; pushIndex < n; pushIndex++) {
            stack.push(pushSequence[pushIndex]);
            while (popIndex < n && !stack.isEmpty() && stack.peek() == popSequence[popIndex]) {
                stack.pop();
                popIndex++;
            }
        }
        return stack.isEmpty();
    }
    */

    /*35.
    输入一个字符串,按字典序打印出该字符串中字符的所有排列。
    例如输入字符串abc,则按字典序打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
    输入字符串长度不超过9(可能有字符重复),字符只包括大小写字母。
    回溯法
    java中String是只读的，没有办法进行变换，因此需要使用StringBuilder。
    TreeSet可避免重复
    private ArrayList<String> res = new ArrayList<>();
    private TreeSet<String> treeSet = new TreeSet<>();
    private StringBuilder stringBuilder = new StringBuilder();
    private boolean[] visited;

    public ArrayList<String> Permutation(String str) {
        if (str == null || str.equals("")) {
            return res;
        }
        char[] strs = str.toCharArray();
        Arrays.sort(strs);//排序
        visited = new boolean[strs.length];
        combination(strs, 0);
        res.addAll(treeSet);//将TreeSet赋给ArrayList
        return res;
    }

    private void combination(char[] strs, int len) {
        if (len == strs.length) {
            treeSet.add(stringBuilder.toString());//结果加到TreeSet
//            其实不使用TreeSet，也可避免重复。只需在向排列集合中增加新排列时，判断其是否已经在集合里。
//            if(!res.contains(stringBuilder.toString()))
//                res.add(stringBuilder.toString());
            return;//回溯
        }
        for (int i = 0; i < strs.length; i++) {
            if (!visited[i]) {
                visited[i] = true;
                stringBuilder.append(strs[i]);
                combination(strs, len + 1);
                //Duang ~ 回溯 - 状态重置
                visited[i] = false;
                stringBuilder.deleteCharAt(stringBuilder.length() - 1);
            }
        }
    }
    */
    public static class TreeNode {
        int val;
        TreeNode left = null;
        TreeNode right = null;
        public TreeNode(int val) {
            this.val = val;
        }
    }
    /*36.
    从上往下打印二叉树
    从上往下打印出二叉树的每个节点，同层节点从左至右打印。
    队列：尾插入，头删除。
    queue函数：容量不够或队列为空时不会抛异常：offer（添加队尾元素）、peek（访问队头元素）、poll（访问队头元素并移除）
              容量不够或队列为空时抛异常：add、element（访问队列元素）、remove（访问队头元素并移除）
    public static ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        if(root == null)
            return result;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);//添加队尾元素
        while(!queue.isEmpty()){
            TreeNode temp = queue.poll();//访问队头元素并移除
            result.add(temp.val);
            if(temp.left != null)
                queue.offer(temp.left);
            if(temp.right != null)
                queue.offer(temp.right);
        }
        return result;
    }
     */
    /*37.
    把二叉树打印成多行
    在上题基础上，将节点数按层保存
    public static ArrayList<ArrayList<Integer>> PrintFromTopToBottom(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if(root == null)
            return result;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);//添加队尾元素
        while(!queue.isEmpty()){
            int size=queue.size();//每次循环，size都为该层节点数
            ArrayList<Integer>list=new ArrayList<Integer>();
            for(int i=0;i<size;i++){
                TreeNode temp = queue.poll();//访问队头元素并移除
                list.add(temp.val);
                if(temp.left != null)
                    queue.offer(temp.left);
                if(temp.right != null)
                    queue.offer(temp.right);
            }
            result.add(list);
        }
        return result;
    }
     */
    /*38.
    按之字形顺序打印二叉树
    请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
    在上题基础上加一个逆序判断
    public static ArrayList<ArrayList<Integer>> PrintFromTopToBottom(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if(root == null)
            return result;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);//添加队尾元素
        boolean reverse=false;
        while(!queue.isEmpty()){
            int size=queue.size();//每次循环，size都为该层节点数
            ArrayList<Integer>list=new ArrayList<Integer>();
            for(int i=0;i<size;i++){
                TreeNode temp = queue.poll();//访问队头元素并移除
                list.add(temp.val);
                if(temp.left != null)
                    queue.offer(temp.left);
                if(temp.right != null)
                    queue.offer(temp.right);
            }
            if(reverse)
                Collections.reverse(list);
            result.add(list);
            reverse=!reverse;
        }
        return result;
    }
    */
    /*39.
    二叉搜索树的后序遍历序列
    输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。假设输入的数组的任意两个数字都互不相同。
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0)
            return false;
        return verify(sequence, 0, sequence.length - 1);
    }
    private boolean verify(int[] sequence, int first, int last) {
        if (last - first <= 1)//遍历结束
            return true;
        int rootVal = sequence[last];//根节点
        int cutIndex = first;
        while (cutIndex < last && sequence[cutIndex] <= rootVal)//左子树，节点值小于根节点
            cutIndex++;
        for (int i = cutIndex; i < last; i++)
            if (sequence[i] < rootVal)//右子树节点值小于根节点，不符合二叉搜索树条件
                return false;
        return verify(sequence, first, cutIndex - 1) && verify(sequence, cutIndex, last - 1);//分别对左右子树递归判断
    }
    */
    /*41.
    复杂链表的复制
    输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的 head。
    public class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;
        RandomListNode(int label) {
            this.label = label;
        }
    }
    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null)
            return null;
        // 插入新节点
        RandomListNode cur = pHead;
        while (cur != null) {
            RandomListNode clone = new RandomListNode(cur.label);
            clone.next = cur.next;
            cur.next = clone;
            cur = clone.next;//cur.next.next
        }
        // 建立 random 链接
        cur = pHead;
        while (cur != null) {
            RandomListNode clone = cur.next;
            if (cur.random != null)
                clone.random = cur.random.next;
            cur = clone.next;//cur.next.next
        }
        // 拆分
        cur = pHead;
        RandomListNode pCloneHead = pHead.next;
        while (cur.next != null) {//依次改变next链接
            RandomListNode tmp = cur.next;
            cur.next = tmp.next;
            cur = tmp;
        }
        return pCloneHead;
    }
     */
    /*42.
    二叉搜索树与双向链表
    输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
    （一）中序遍历二叉树，用一个ArrayList类保存遍历的结果，然后再来修改指针
    public TreeNode Convert(TreeNode pRootOfTree) {
        if(pRootOfTree == null){
            return null;
        }
        ArrayList<TreeNode> list = new ArrayList<>();
        Convert(pRootOfTree, list);
        return Convert(list);

    }
    //中序遍历，在list中按遍历顺序保存
    public void Convert(TreeNode pRootOfTree, ArrayList<TreeNode> list){
        if(pRootOfTree.left != null){
            Convert(pRootOfTree.left, list);
        }

        list.add(pRootOfTree);

        if(pRootOfTree.right != null){
            Convert(pRootOfTree.right, list);
        }
    }
    //遍历list，修改指针
    public TreeNode Convert(ArrayList<TreeNode> list){
        for(int i = 0; i < list.size() - 1; i++){
            list.get(i).right = list.get(i + 1);
            list.get(i + 1).left = list.get(i);
        }
        return list.get(0);
    }
    （二）分治：
    private TreeNode pLast = null;//记录当前链表的末尾节点
    public TreeNode Convert(TreeNode root) {
        if (root == null)
            return null;
        // 先令左子树形成链表
        // 如果左子树为空，那么根节点root为双向链表的头节点
        TreeNode head = Convert(root.left);
        if (head == null)
            head = root;
        // 连接当前节点root和当前链表的尾节点pLast
        root.left = pLast;
        if (pLast != null)
            pLast.right = root;
        pLast = root;
        //再把右子树也加到链表里
        Convert(root.right);
        return head;
    }
     */

    /*44.
    数组中出现次数超过一半的数字
    数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。（假设符合条件的数字一定存在）
    多数投票问题，可以利用 Boyer-Moore Majority Vote Algorithm 来解决这个问题，使得时间复杂度为 O(N)。
    使用 cnt 来统计一个元素出现的次数，当遍历到的元素和统计元素相等时，令 cnt++，否则令 cnt--。
    如果前面查找了 i个元素，且 cnt == 0，说明前 i 个元素没有 majority，或者有 majority，但是出现的次数少于 i / 2 ，
    因为如果多于 i / 2的话 cnt 就一定不会为 0 。因为+1的次数会比-1的次数多。
    此时剩下的 n - i 个元素中，majority 的数目依然多于 (n - i) / 2，因此继续查找就能找出majority。
    //[1,2,3,2,2,2,5,4,2]  2
    public int MoreThanHalfNum_Solution(int[] nums) {
        int majority = 0;
        int count = 0;
        for (int i : nums) {
            if (count == 0) {
                majority = i;
                count++;
            } else if (i == majority) {
                count++;
            } else {
                count--;
            }
        }
        return majority;
    }
     */
    /*45.
    最小的 K 个数
    给定一个数组，找出其中最小的K个数。
    快速选择：
    复杂度：O(N) + O(1)
    只有当允许修改数组元素时才可以使用
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] nums, int k) {
        ArrayList<Integer> ret = new ArrayList<>();
        if (k > nums.length || k <= 0)
            return ret;
        findKthSmallest(nums, k - 1);
        for (int i = 0; i < k; i++)
            ret.add(nums[i]);
        return ret;
    }
    public void findKthSmallest(int[] nums, int k) {
        int l = 0, h = nums.length - 1;
        while (l < h) {
            int j = partition(nums, l, h);
            if (j == k)
                break;
            if (j > k)
                h = j - 1;
            else
                l = j + 1;
        }
    }
    //快排
    // 返回一个整数 j 使得 a[l..j-1] 小于等于 a[j]，且 a[j+1..h] 大于等于 a[j]，此时 a[j] 就是数组的第 j 大元素。
    private int partition(int[] nums, int l, int h) {
        int p = nums[l]; // 切分元素
        int i = l, j = h + 1;
        while (true) {
        while (i != h && nums[++i] < p) ;
        while (j != l && nums[--j] > p) ;
        if (i >= j)
            break;
        swap(nums, i, j);
        }
        swap(nums, l, j);
        return j;
    }
    private void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }
    大小为 K 的最小堆
    复杂度：O(NlogK) + O(K)
    特别适合处理海量数据
    大顶堆：每个结点的值都大于或等于其左右孩子结点的值
    小顶堆：每个结点的值都小于或等于其左右孩子结点的值
    应该使用大顶堆来维护最小堆，而不能直接创建一个小顶堆并设置一个大小，企图让小顶堆中的元素都是最小元素。
    （因为大顶堆只需要与堆顶比较，小顶堆需要跟“叶子”比较）
    维护一个大小为 K 的最小堆过程如下：在添加一个元素之后，如果大顶堆的大小大于 K，那么需要将大顶堆的堆顶元素去除。
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] nums, int k) {
        if (k > nums.length || k <= 0)
            return new ArrayList<>();
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((o1, o2) -> o2 - o1);//指定为大顶堆
        for (int num : nums) {
            maxHeap.add(num);
            if (maxHeap.size() > k)
                maxHeap.poll();
        }
        return new ArrayList<>(maxHeap);
    }
     */

    /*46.
    数据流中的中位数
    如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
    如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
    //大顶堆，存储左半边元素
    private PriorityQueue<Integer> left = new PriorityQueue<>((o1, o2) -> o2 - o1);
    // 小顶堆，存储右半边元素，并且右半边元素都大于左半边
    private PriorityQueue<Integer> right = new PriorityQueue<>();
    // 当前数据流读入的元素个数
    private int N = 0;
    public void Insert(Integer val) {
        // 插入要保证两个堆存于平衡状态
        if (N % 2 == 0) {
//            N 为偶数的情况下插入到右半边。
//            因为右半边元素都要大于左半边，但是新插入的元素不一定比左半边元素来的大，
//            因此需要先将元素插入左半边，然后利用左半边为大顶堆的特点，取出堆顶元素即为最大元素，此时插入右半边
            left.add(val);
            right.add(left.poll());//大顶堆取最大
        } else {
            //同理，N 为奇数的情况下插入到左半边。
            right.add(val);
            left.add(right.poll());//小顶堆取最小
        }
        N++;
    }
    public Double GetMedian() {
        if (N % 2 == 0)
            return (left.peek() + right.peek()) / 2.0;
        else
            return (double) right.peek();
    }
     */
    /*47.
    字符流中第一个不重复的字符
    请实现一个函数用来找出字符流中第一个只出现一次的字符。
    例如，当从字符流中只读出前两个字符 "go" 时，第一个只出现一次的字符是 "g"。
    当从该字符流中读出前六个字符“google" 时，第一个只出现一次的字符是 "l"。
    不重复：hash
    第一个：先进先出->queue
    //（一）
    private int[] charCnt = new int[128];//ASCII码数量为128
    private Queue<Character> queue = new LinkedList<Character>();
    public void Insert(char ch) {
        //System.out.println(ch+":"+charCnt[ch]);
        if (charCnt[ch]++ == 0)  //新的不重复的字符，入队;出现过的重复字符只charCnt++，不入队
            queue.add(ch);
        //System.out.println(ch+"---:"+charCnt[ch]);
    }
    public char FirstAppearingOnce() {
        Character CHAR = null;
        char c = 0;
        while ((CHAR = queue.peek()) != null) {//访问队头元素
            c = CHAR.charValue();
            if (charCnt[c] == 1) //未重复则输出
                return c;
            else queue.remove(); //重复则移出队列
        }
        return '#'; //队空，返回#
    }
    //（二）
    private int[] cnts = new int[128];
    private Queue<Character> queue = new LinkedList<>();
    public void Insert(char ch) {
        cnts[ch]++;
        queue.add(ch);
        while (!queue.isEmpty() && cnts[queue.peek()] > 1)
            queue.poll();
    }
    public char FirstAppearingOnce() {
        return queue.isEmpty() ? '#' : queue.peek();
    }
     */

    /*48.
    连续子数组的最大和
    输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。要求时间复杂度为 O(n).
    输入的数组为{1,-2,3,10,—4,7,2,一5}，和最大的子数组为{3,10,一4,7,2}，因此输出为该子数组的和 18。
    public int FindGreatestSumOfSubArray(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int greatestSum = Integer.MIN_VALUE;
        int sum = 0;
        for (int val : nums) {
            sum = sum <= 0 ? val : sum + val;//当和一旦为负数，则舍弃该组数字
            greatestSum = Math.max(greatestSum, sum);//保证greatestSum始终为最大和
        }
        return greatestSum;
    }
     */
    /*49.
    整数中 1 出现的次数（从 1 到 n ）
    输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。例如，1~13中包含1的数字有1、10、11、12、13因此共出现6次。
    参考：https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/mian-shi-ti-43-1n-zheng-shu-zhong-1-chu-xian-de-2/
    public int NumberOf1Between1AndN_Solution(int n) {
        int digit = 1, res = 0;
        int high = n / 10, cur = n % 10, low = 0;
        while(high != 0 || cur != 0) {
            if (cur == 0) res += high * digit;
            else if (cur == 1) res += high * digit + low + 1;
            else res += (high + 1) * digit;
            low += cur * digit;
            cur = high % 10;
            high /= 10;
            digit *= 10;
        }
            return res;
    }
    */

    /*50.
    数字序列中某一位数字
    数字以 0123456789101112131415... 的格式序列化到一个字符串中，求这个字符串的第 index 位。
//    　　观察规律：
//            　　个位数的个数一共有10个，即0~9，共占了10*1位数字；
//            　　两位数的个数一共有90个，即10~99，每个数字占两位，共占了90*2位数字；
//            　　……
//            　　m位数的个数一共有9*10^(m-1)个，每个数字占m位，占了9*10^(m-1)*m位数字。
    public int digitAtIndex(int index) {
        if(index<0)
            return -1;
        int m=1;  //m位数
        while(true) {
            int numbers=numbersOfIntegers(m);  //m位数的个数
            if(index<numbers*m)
                return getDigit(index,m);
            index-=numbers*m;//***
            m++;
        }
    }
    //返回m位数的总个数
    //例如，两位数一共有90个：10~99；三位数有900个：100~999
    private int numbersOfIntegers(int m) {
        if(m==1)
            return 10;
        return (int) (9*Math.pow(10, m-1));
    }
    //获取数字
    private int getDigit(int index, int m) {
        System.out.println("index:"+index);
        System.out.println("m:"+m);
        int number=getFirstNumber(m)+index/m;  //对应的m位数
        System.out.println("number:"+number);
        int indexFromRight = m-index%m;  //在数字中的位置
        System.out.println("indexFromRight:"+indexFromRight);
        for(int i=1;i<indexFromRight;i++)
            number/=10;
        return number%10;
    }
    //第一个m位数
    //例如第一个两位数是10，第一个三位数是100
    private int getFirstNumber(int m) {
        if(m==1)
            return 0;
        return (int) Math.pow(10, m-1);
    }
     */
    /*51.
    把数组排成最小的数
    输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
    例如输入数组 {3，32，321}，则打印出这三个数字能排成的最小数字为 321323。
    可以看成是一个排序问题，在比较两个字符串 S1 和 S2 的大小时，应该比较的是 S1+S2 和 S2+S1 的大小，
    如果 S1+S2< S2+S1，那么应该把 S1 排在前面，否则应该把 S2 排在前面。
    public String PrintMinNumber(int[] numbers) {
        if (numbers == null || numbers.length == 0)
            return "";
        int n = numbers.length;
        String[] nums = new String[n];
        for (int i = 0; i < n; i++)
            nums[i] = numbers[i] + "";//int转String
        Arrays.sort(nums, (s1, s2) -> (s1 + s2).compareTo(s2 + s1));
        String ret = "";
        for (String str : nums)
            ret += str;
        return ret;
    }
     */
    /*52.
    把数字翻译成字符串
    给定一个数字，按照如下规则翻译成字符串：1 翻译成“a”，2 翻译成“b”... 26 翻译成“z”。一个数字有多种翻译可能，
    例如 12258 一共有 5 种，分别是 abbeh，lbeh，aveh，abyh，lyh。实现一个函数，用来计算一个数字有多少种不同的翻译方法。
    public int numDecodings(String s) {
        int n = s.length();
        int[] f = new int[n + 1];
        f[0] = 1;
        for (int i = 1; i <= n; ++i) {
            if (s.charAt(i - 1) != '0') {
                f[i] += f[i - 1];
            }
            if (i > 1 && s.charAt(i - 2) != '0' && ((s.charAt(i - 2) - '0') * 10 + (s.charAt(i - 1) - '0') <= 26)) {
                f[i] += f[i - 2];
            }
        }
        return f[n];
    }
    //进一步节省空间：分别用a,b,c来代替f[i],f[i - 1],f[i - 2]
     */
    /*53.
    礼物的最大价值
    在一个 m*n 的棋盘的每一个格都放有一个礼物，每个礼物都有一定价值（大于 0）。从左上角开始拿礼物，每次向右或向下移动一格，直到右下角结束。
    给定一个棋盘，求拿到礼物的最大价值。
    动态规划
    public int getMost(int[][] values) {
        if (values == null || values.length == 0 || values[0].length == 0)
            return 0;
        int n = values.length;
        int[][] dp = new int[n][n];
        dp[0][0] = values[0][0];
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i-1]+values[0][i];//第一行
            dp[i][0] =dp[i-1][0]+values[i][0];//第一列
        }
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + values[i][j];//动态规划
            }
        }
        return dp[n - 1][n - 1];
    }
     */
    /*54.
    最长不含重复字符的子字符串
    输入一个字符串（只包含 a~z 的字符），求其最长不含重复字符的子字符串的长度。
    例如对于 arabcacfr，最长不含重复字符的子字符串为 rabc或acfr，长度为 4。
    采用动态规划，首先定义函数f(i)表示以第i个字符为结尾的不包含重复字符的子字符串的最长长度,则有以下三种情形：
    ①第i个字符在之前都没有出现过，则f(i) = f(i-1)+1
    ②第i个字符在之前出现过，但是在f(i-1)这个子串的前面出现过，则最长还是f(i-1)+1
    ③第i个字符在之前出现过，不过在f(i-1)的这个子串的中间出现过，则f(i)=这两个重复字符的中间值
    public int longestSubStringWithoutDuplication(String str) {
        int curLen = 0;//当前长度
        int maxLen = 0;//最大长度
        int[] preIndexs = new int[26];//保存a-z字符出现位置
        Arrays.fill(preIndexs, -1);//-1表示没出现过
        for (int i = 0; i < str.length(); i++) {
            int c = str.charAt(i) - 'a';
            int preI = preIndexs[c];
            if (preI == -1 || i - preI > curLen) {//没出现过或在f(i-1)子串的前面出现
                curLen++;
            } else {//在f(i-1)子串中出现
                maxLen = Math.max(maxLen, curLen);//此时需要对字符串“截断”，因此需要先保留此时的长度
                curLen = i - preI;
            }
            preIndexs[c] = i;//更新当前字符最近一次的出现位置
        }
        maxLen = Math.max(maxLen, curLen);
        return maxLen;
    }
     */
    /*55.
    丑数
    把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。例如 6、8 都是丑数，但 14 不是，因为它包含因子 7。
    习惯上我们把 1 当做是第一个丑数。求按从小到大的顺序的第 N 个丑数。
    本题自己是有思路的，丑数能够分解成2x3y5z,所以只需要把得到的丑数不断地乘以2、3、5之后并放入他们应该放置的位置即可。
    而难点就在于如何有序的放在合适的位置。
    1乘以 （2、3、5）=2、3、5；2乘以（2、3、5）=4、6、10；3乘以（2、3、5）=6,9,15；5乘以（2、3、5）=10、15、25；
    从这里我们可以看到如果不加策略地添加丑数是会有重复并且无序，
    而在2x，3y，5z中，如果x=y=z那么最小丑数一定是乘以2的，但关键是有可能存在x》y》z的情况，
    所以我们要维持三个指针来记录当前乘以2、乘以3、乘以5的最小值，
    当其被选为新的最小值后，要把相应的指针+1；
    因为这个指针会逐渐遍历整个数组，因此最终数组中的每一个值都会被乘以2、乘以3、乘以5，
    也就是实现了我们最开始的想法，只不过不是同时成乘以2、3、5，而是在需要的时候乘以2、3、5.
    public int GetUglyNumber_Solution(int N) {
        if (N <= 6)
            return N;
        int i2 = 0, i3 = 0, i5 = 0;//因子2、3和5的数量，也是指针
        int[] dp = new int[N];
        dp[0] = 1;
        for (int i = 1; i < N; i++) {
            int next2 = dp[i2] * 2, next3 = dp[i3] * 3, next5 = dp[i5] * 5;//即x、y、z不一定相等
            dp[i] = Math.min(next2, Math.min(next3, next5));//最小值，保证从小到大的顺序
            if (dp[i] == next2)//为了防止重复，三个if都得能走到
                i2++;
            if (dp[i] == next3)
                i3++;
            if (dp[i] == next5)
                i5++;
        }
        return dp[N - 1];
    }
     */
    /*56.
    第一个只出现一次的字符位置
    在一个字符串中找到第一个只出现一次的字符，并返回它的位置。
    最直观的解法是使用 HashMap 对出现次数进行统计，但是考虑到要统计的字符范围有限，因此可以使用整型数组代替 HashMap。
    public int FirstNotRepeatingChar(String str) {
        int[] cnts = new int[128];
        for (int i = 0; i < str.length(); i++)
            cnts[str.charAt(i)]++;
        for (int i = 0; i < str.length(); i++)
            if (cnts[str.charAt(i)] == 1)
                return i;
        return -1;
    }
    // bs2 bs1 两个比特位表示出现0次、1次和多次（00,01,11）
    public int FirstNotRepeatingChar(String str) {
        BitSet bs1 = new BitSet(128);
        BitSet bs2 = new BitSet(128);
        for (char c : str.toCharArray()) {
            if (!bs1.get(c) && !bs2.get(c))//第一次出现
                bs1.set(c); // 0 0 -> 0 1
            else if (bs1.get(c) && !bs2.get(c))//出现过一次的又出现
                bs2.set(c); // 0 1 -> 1 1
        }//已经出现过两次以上（11）的不需要再操作
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (bs1.get(c) && !bs2.get(c)) // 0 1
                return i;//找到第一个则返回
        }
        return -1;
    }
     */
    /*57.
    数组中的逆序对
    在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数（对1000000007取模）。
    归并排序：
    ①递归划分整个区间为基本相等的左右两个区间；
    ②合并两个有序区间
    private long cnt = 0;
    private int[] tmp; // 在这里声明辅助数组，而不是在 merge()递归函数中声明，节省花销
    public int InversePairs(int[] nums) {
        tmp = new int[nums.length];
        mergeSort(nums, 0, nums.length - 1);
        return (int) (cnt % 1000000007);
    }
    private void mergeSort(int[] nums, int l, int h) {
        if (h - l < 1)
            return;
        int m = l + (h - l) / 2;
        mergeSort(nums, l, m);
        mergeSort(nums, m + 1, h);
        merge(nums, l, m, h);
    }
    private void merge(int[] nums, int low, int middle, int high) {
        int i = low, j = middle + 1, k = low;
        while (i <= middle || j <= high) {
            if (i > middle)//j <= high
                tmp[k] = nums[j++];
            else if (j > high)//i <= middle
                tmp[k] = nums[i++];
            else if (nums[i] <= nums[j])
                tmp[k] = nums[i++];
            else {//逆序，nums[i] > nums[j]
                tmp[k] = nums[j++];
                this.cnt += middle - i + 1; //有序区间，nums[i] > nums[j]，说明 nums[i...mid] 都大于 nums[j]
            }
            k++;
        }
        for (int q = low; q <= high; q++)
            nums[q] = tmp[q];//修改nums
    }
     */

    /*58.
    两个链表的第一个公共结点
    设 A 的长度为 a + c，B 的长度为 b + c，其中 c 为尾部公共部分长度，可知 a + c + b = b + c + a。
    当访问链表 A 的指针访问到链表尾部时，令它从链表 B 的头部重新开始访问链表 B；
    同样地，当访问链表 B 的指针访问到链表尾部时，令它从链表 A 的头部重新开始访问链表 A。
    这样就能控制访问 A 和 B 两个链表的指针能同时访问到交点。
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode l1 = pHead1, l2 = pHead2;
        while (l1 != l2) {
            l1 = (l1 == null) ? pHead2 : l1.next;
            l2 = (l2 == null) ? pHead1 : l2.next;
        }
        return l1;
    }
     */
    /*59.
    数字在排序数组中出现的次数
    Input:
        nums = 1, 2, 3, 3, 3, 3, 4, 6
        K = 3
    Output:4
    排序数组，则只需找到数字第一次出现的位置和下一个数字第一次出现的位置，做个减法即可
    public int GetNumberOfK(int[] nums, int K) {
        int first = binarySearch(nums, K);
        int last = binarySearch(nums, K + 1);//k+1即使不存在，也会指向大于k的第一个位置
        return (first == nums.length || nums[first] != K) ? 0 : last - first;//不存在则指向数组末尾或下一数字位置
    }
    private int binarySearch(int[] nums, int K) {
        int l = 0, h = nums.length;
        while (l < h) {//二分查找
            int m = l + (h - l) / 2;
            if (nums[m] >= K)
                h = m;
            else
                l = m + 1;//下一位置
        }
        return l;
    }
    //其实直接遍历也很快的
     */
    /*60.
    二叉查找树的第 K 个结点
    给定一棵二叉搜索树，请找出其中的第k小的TreeNode结点。
    利用二叉查找树中序遍历有序的特性
    private TreeNode ret;
    private int cnt = 0;
    public TreeNode KthNode(TreeNode pRoot, int k) {
        inOrder(pRoot, k);
        return ret;
    }
    private void inOrder(TreeNode root, int k) {
        if (root == null || cnt >= k)
            return;
        inOrder(root.left, k);
        cnt++;
        if (cnt == k)
            ret = root;
        inOrder(root.right, k);
    }
     */
    /*61.
    二叉树的深度
    从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
    public int TreeDepth(TreeNode root) {
        return root == null ? 0 : 1 + Math.max(TreeDepth(root.left), TreeDepth(root.right));
    }
    */
    /*62.
    平衡二叉树
    输入一棵二叉树，判断该二叉树是否是平衡二叉树。
    在这里，我们只需要考虑其平衡性，不需要考虑其是不是排序二叉树
    平衡二叉树（Balanced Binary Tree），具有以下性质：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，
    并且左右两个子树都是一棵平衡二叉树。
    注：我们约定空树是平衡二叉树。
    private boolean isBalanced = true;
    public boolean IsBalanced_Solution(TreeNode root) {
        height(root);
        return isBalanced;
    }
    private int height(TreeNode root) {
        if (root == null || !isBalanced)
            return 0;
        int left = height(root.left);
        int right = height(root.right);
        if (Math.abs(left - right) > 1)
            isBalanced = false;
        return 1 + Math.max(left, right);
    }
     */
    /*63.
    数组中只出现一次的数字
    一个整型数组里除了两个数字之外，其他的数字都出现了两次，找出这两个数。
    两个不相等的元素在位级表示上必定会有一位存在不同，将数组的所有元素异或得到的结果为不存在重复的两个元素异或的结果。
    diff&=-diff得到diff最右侧不为 0 的位，也就是不存在重复的两个元素在位级表示上最右侧不同的那一位，利用这一位就可以将两个元素区分开来。
    ^异或（相同为0，不同为1）
    负数二进制为补码（先取反得反码，再末位取反），&逻辑与（11为1,其余为0）
    //{1,3,4,5,1,5,6,6}
    public static void FindNumsAppearOnce(int[] nums, int[] num1, int[] num2) {
        int diff = 0;
        //System.out.print("diff:");
        for (int num : nums){
            diff ^= num;//得到两个单身元素不相同的位数，例如7，即111，则表示3位都不相同
            //System.out.print(diff+" ");
        }
        //System.out.println();
        diff &= -diff;//找到两个单身元素不相同的位数之一（最右一个），根据该位即可区分两个元素（不是单身的元素必定在同一组里）
        //System.out.println("diff:"+diff);
        for (int num : nums) {
            if ((num & diff) == 0)//分两组
                num1[0] ^= num;//异或可以去掉不是单身的元素
            else
                num2[0] ^= num;
            //System.out.print(num1+" ");
            //System.out.print(num2+" ");
        }
    }
    */
    /*64.
    和为 S 的两个数字
    输入一个递增排序的数组和一个数字 S，在数组中查找两个数，使得他们的和正好是 S。如果有多对数字的和等于S，输出两个数的乘积最小的。
    最外层的乘积最小!!!!!
    使用双指针，一个指针指向元素较小的值，一个指针指向元素较大的值。指向较小元素的指针从头向尾遍历，指向较大元素的指针从尾向头遍历。
    如果两个指针指向元素的和 sum == target，那么得到要求的结果；
        如果 sum > target，移动较大的元素，使 sum 变小一些；
        如果 sum < target，移动较小的元素，使 sum 变大一些
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        int i = 0, j = array.length - 1;
        while (i < j) {
            int cur = array[i] + array[j];
            if (cur == sum)
                return new ArrayList<>(Arrays.asList(array[i], array[j]));
            if (cur < sum)
                i++;
            else
                j--;
        }
        return new ArrayList<>();
    }
    */
    /*65.
    和为 S 的连续正数序列
    例如S=100:
        [9, 10, 11, 12, 13, 14, 15, 16]
        [18, 19, 20, 21, 22]
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        int start = 1, end = 2;
        int curSum = 3;
        while (end < sum) {
            if (curSum > sum) {
                curSum -= start;//从头开始往下去，保持序列的连续性
                start++;
            } else if (curSum < sum) {
                end++;
                curSum += end;
            } else {//保存结果并调整curSum继续寻找
                ArrayList<Integer> list = new ArrayList<>();
                for (int i = start; i <= end; i++)
                    list.add(i);
                ret.add(list);
                curSum -= start;
                start++;
                end++;
                curSum += end;
            }
        }
        return ret;
    }
     */
    /*66.
    翻转单词顺序列
    Input:"student. a am I"
    Output:"I am a student."
    public static String ReverseSentence(String str) {
        String[]chars=str.split(" ");
        String []rets=new String[chars.length];
        for (int i=0,j=chars.length-1;i<chars.length;i++,j--){
            rets[i]=chars[j];
        }
        String rr="";
        for (int i=0;i<chars.length-1;i++){
            rr+=rets[i]+" ";
        }
        rr+=rets[chars.length-1];
        return rr;
    }
    限制在O(n)即一个字符数组的空间复杂度：
    public String ReverseSentence(String str) {
        int n = str.length();
        char[] chars = str.toCharArray();//按字符分割
        int i = 0, j = 0;
        while (j <= n) {
            if (j == n || chars[j] == ' ') {//翻转单词
                reverse(chars, i, j - 1);
                i = j + 1;
            }
            j++;
        }
//        for(char c:chars)
//            System.out.print(c);
//        System.out.println();
        reverse(chars, 0, n - 1);//翻转整个字符串
        //翻转两词，保持字母正序（先倒序再正序）
        return new String(chars);
    }
    private void reverse(char[] c, int i, int j) {
        while (i < j)
            swap(c, i++, j--);
    }
    private void swap(char[] c, int i, int j) {
        char t = c[i];
        c[i] = c[j];
        c[j] = t;
    }
     */
    /*67.
    左旋转字符串
    Input:
        S="abcXYZdef"
        K=3
    Output:
    "XYZdefabc"
    先将 "abc" 和 "XYZdef" 分别翻转，得到 "cbafedZYX"，然后再把整个字符串翻转得到 "XYZdefabc"。
    public String LeftRotateString(String str, int n) {
        if (n >= str.length())
            return str;
        char[] chars = str.toCharArray();
        reverse(chars, 0, n - 1);
        reverse(chars, n, chars.length - 1);
        reverse(chars, 0, chars.length - 1);
        return new String(chars);
    }
    private void reverse(char[] chars, int i, int j) {
        while (i < j)
            swap(chars, i++, j--);
    }
    private void swap(char[] chars, int i, int j) {
        char t = chars[i];
        chars[i] = chars[j];
        chars[j] = t;
    }
    */
    /*68.
    滑动窗口的最大值
    给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
    例如，如果输入数组 {2, 3, 4, 2, 6, 2, 5, 1} 及滑动窗口的大小 3，那么一共存在 6 个滑动窗口，他们的最大值分别为 {4,4, 6, 6, 6, 5}。
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> ret = new ArrayList<>();
        if (size > num.length || size < 1)
            return ret;
        PriorityQueue<Integer> heap = new PriorityQueue<>((o1, o2) -> o2 - o1); // 大顶堆
        for (int i = 0; i < size; i++)
            heap.add(num[i]);
        ret.add(heap.peek());//访问堆头
        for (int i = 0, j = i + size; j < num.length; i++, j++) { // 维护一个大小为 size 的大顶堆
            heap.remove(num[i]);//移除相应元素
            heap.add(num[j]);
            ret.add(heap.peek());
        }
        return ret;
    }
     */
    /*69.
    n 个骰子的点数
    把 n 个骰子仍在地上，求点数和为 s 的概率。
    动态规划解法：使用一个二维数组 dp 存储点数出现的次数，其中 dp[i][j] 表示前 i 个骰子产生点数 j 的次数。空间复杂度：O(N^2)
    dp[i][j]=dp[i-1][j-1]+dp[i-1][j-2]+...+dp[i-1][j-6]（0<j<=6i）
    dp[i][j]=0 (j>6i或者j<i)
    public List<Map.Entry<Integer, Double>> dicesSum(int n) {
        final int face = 6;
        final int pointNum = face * n;
        long[][] dp = new long[n + 1][pointNum + 1];
        for (int i = 1; i <= face; i++)
            dp[1][i] = 1;//1个骰子，出现点数1-6的次数均为 1
        for (int i = 2; i <= n; i++)
            for (int j = i; j <= pointNum; j++) // 使用 i 个骰子最小点数为 i
                for (int k = 1; k <= face; k++) {
                    //System.out.println("j:"+j+",k:"+k+",j-k="+(j-k));
                    if(j>k)
                        dp[i][j] += dp[i - 1][j - k];
                }
        final double totalNum = Math.pow(6, n);//n个骰子每个都有6种可能，则总共6的n方种可能
        List<Map.Entry<Integer, Double>> ret = new ArrayList<>();
        for (int i = n; i <= pointNum; i++)// 使用 n 个骰子最小点数为 n
            ret.add(new AbstractMap.SimpleEntry<>(i, dp[n][i] / totalNum));
        return ret;
    }
     */
    /*70.
    扑克牌顺子
    五张牌，其中大小鬼为癞子，牌面大小为 0（可抵任意牌）。判断这五张牌是否能组成顺子(连续牌)。
    public boolean isContinuous(int[] nums) {
        if (nums.length < 5)
            return false;
        Arrays.sort(nums);
        // 统计癞子数量
        int cnt = 0;
        for (int num : nums)
            if (num == 0)
                cnt++;
        // 使用癞子去补全不连续的顺子
        for (int i = cnt; i < nums.length - 1; i++) {
            if (nums[i + 1] == nums[i])//有相同的则一定不会组成顺子
                return false;
            cnt -= nums[i + 1] - nums[i] - 1;//相邻牌不占用癞子
        }
        System.out.println(cnt);
        return cnt >= 0;//癞子够补全则>=0
    }
     */
    /*71.
    圆圈中最后剩下的数
    让n个小朋友们围成一个大圈。然后，随机指定一个数 m，让编号为 0 的小朋友开始报数。每次喊到 m-1 的那个小朋友要出列唱首歌，
    然后可以在礼品箱中任意的挑选礼物，并且不再回到圈中，从他的下一个小朋友开始，继续 0...m-1报数 .... 这样下去 ....
    直到剩下最后一个小朋友，可以不用表演。
    令f[i]表示i个人时最后胜利者的编号，则有递推公式：
        f[1]=0;
        f[i]=(f[i-1]+m)%i; (i>1)
    约瑟夫环，圆圈长度为 n 的解可以看成长度为 n-1 的解再加上报数的长度 m。因为是圆圈，所以最后需要对 n 取余。
    public int LastRemaining_Solution(int n, int m) {
        if (n <= 0||m<=0)
            return -1;
        if (n == 1)
            return 0;
        return (LastRemaining_Solution(n - 1, m) + m) % n;
    }
     */
    /*72.
    求 1+2+3+...+n
    要求不能使用乘除法、for、while、if、else、switch、case 等关键字及条件判断语句 A ? B : C。
    public int Sum_Solution(int n) {
        int sum = n;
        boolean b = (n > 0) && ((sum += Sum_Solution(n - 1)) > 0);
        return sum;
    }
//    使用递归解法最重要的是指定返回条件，但是本题无法直接使用 if 语句来指定返回条件。
//    条件与 && 具有短路原则，即在第一个条件语句为 false 的情况下不会去执行第二个条件语句。
//    利用这一特性，将递归的返回条件取非然后作为 && 的第一个条件语句，递归的主体转换为第二个条件语句，
//    那么当递归的返回条件为true 的情况下就不会执行递归的主体部分，递归返回。
//    本题的递归返回条件为 n <= 0，取非后就是 n > 0；递归的主体部分为 sum += Sum_Solution(n - 1)，
//    转换为条件语句后就是 (sum += Sum_Solution(n - 1)) > 0。
    */
    /*73.
    不用加减乘除做加法
    写一个函数，求两个整数之和，要求不得使用 +、-、*、/ 四则运算符号
    a ^ b 表示没有考虑进位的情况下两数的和，(a & b) << 1 就是进位。
    递归会终止的原因是 (a & b) << 1 最右边会多一个 0，那么继续递归，进位最右边的 0 会慢慢增多，最后进位会变为0，递归终止。
    public int Add(int a, int b) {
        return b == 0 ? a : Add(a ^ b, (a & b) << 1);
    }
     */
    /*74.
    构建乘积数组
    给定一个数组 A[0, 1,..., n-1]，请构建一个数组 B[0, 1,..., n-1]，
    其中 B 中的元素 B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。要求不能使用除法。
    public int[] multiply(int[] A) {
        if(A==null || A.length<2)
            return null;
        int[] B=new int[A.length];
        B[0]=1;
        for(int i=1;i<A.length;i++)//从左往右
            B[i]=B[i-1]*A[i-1];//left[i]=left[i-1]*A[i-1]
        //循环结束，B数组中暂时存放了自己对应左下三角的乘积
        int temp=1;
        for(int i=A.length-2;i>=0;i--){//从右往左
            temp*=A[i+1];//right[i]=right[i+1]*A[i+1]
            B[i]*=temp;//left[i]*right[i]
        }
        return B;
    }
     */
    /*75.
    把字符串转换成整数
    将一个字符串转换成一个整数，字符串不是一个合法的数值则返回 0，要求不能使用字符串转换整数的库函数。
    public int StrToInt(String str) {
        if (str == null || str.length() == 0)
            return 0;
        boolean isNegative = str.charAt(0) == '-';
        int ret = 0;
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (i == 0 && (c == '+' || c == '-')) // 符号判定
                continue;
            if (c < '0' || c > '9') // 非法输入
                return 0;
            ret = ret * 10 + (c - '0');
        }
        return isNegative ? -ret : ret;
    }
     */
    /*76.
    树中两个节点的最低公共祖先
    二叉查找树中，两个节点 p, q 的最低公共祖先 root 满足 root.val >= p.val && root.val <= q.val
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null)
            return null;
        if (root.val > p.val && root.val > q.val)
            return lowestCommonAncestor(root.left, p, q);
        if (root.val < p.val && root.val < q.val)
            return lowestCommonAncestor(root.right, p, q);
        return root;
    }
    普通二叉树：在左右子树中查找是否存在 p 或者 q，如果 p 和 q 分别在两个子树中，那么就说明根节点就是最低公共祖先。
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q)
            return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        //return left == null ? right : right == null ? left : root;
        if(left == null)
            return right;
        else
            if(right == null)
                return left;
            else
                return root;
    }
    写法二（配套图解）：
    时间复杂度：O(N)，其中 N 是二叉树的节点数。二叉树的所有节点有且只会被访问一次，因此时间复杂度为 O(N)。
    空间复杂度：O(N) ，其中 N 是二叉树的节点数。递归调用的栈深度取决于二叉树的高度，
              二叉树最坏情况下为一条链，此时高度为 N，因此空间复杂度为 O(N)。
    private TreeNode ans;
    private boolean dfs(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return false;
        boolean lson = dfs(root.left, p, q);
        boolean rson = dfs(root.right, p, q);
        if ((lson && rson) || ((root.val == p.val || root.val == q.val) && (lson || rson))) {
            ans = root;
        }
        return lson || rson || (root.val == p.val || root.val == q.val);
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        this.dfs(root, p, q);
        return this.ans;
    }
    */






















        public static void main(String[] args) {
        /*1.
        int nums[]={2, 3, 1, 0, 2, 5};
        int duplication=duplicate(nums,6);
        System.out.println(duplication);*/
        /*2.
        int[][] nums = new int[][]{{1,4,7,11,15},{2,5,8,12,19},{3,6,9,16,22},{10,13,14,17,24},{18,21,23,26,30}};
        int target1=5,target2=20;
        System.out.println(Find(target1,nums));
        System.out.println(Find(target2,nums));*/
        /*3.
        StringBuffer str=new StringBuffer("A B");
        System.out.println(replaceSpace(str));*/
        /*4.
        ListNode a=new ListNode(1),b=new ListNode(2),c=new ListNode(3),d=new ListNode(4);
        a.next=b; b.next=c; c.next=d;
        ArrayList<Integer> ret =printListFromTailToHead(a);
        for(int i=0;i<ret.size();i++){
            System.out.println(ret.get(i));
        }*/
        /*5.
        int preorder[] = {1,2,3,4,5,6,7}, inorder[] = {3,2,4,1,6,5,7};
        TreeNode root = reConstructBinaryTree(preorder,inorder);
        PrintFromTopToBottom(root);*/
        /*14.
        Scanner in=new Scanner(System.in);
        String input=in.next();
        String[] inputs = input.split(",") ;
        char[]array=inputs[0].toCharArray();
        int rows =Integer.parseInt(inputs[1]);
        int rols=Integer.parseInt(inputs[2]);
        char[] str=inputs[3].toCharArray();
        Solution s=new Solution();
        System.out.println(s.hasPath(array,rows,rols,str));*/
        /*19.
        Scanner in=new Scanner(System.in);
        String input=in.next();
        int n=Integer.parseInt(input);
        Solution s=new Solution();
        s.print1ToMaxOfNDigits(n); */
        /*20.
        Scanner in=new Scanner(System.in);
        String input=in.next();
        String[] inputs = input.split(",") ;
        char[]str=inputs[0].toCharArray();
        char[]pattern =inputs[1].toCharArray();
        Solution s = new Solution();
        System.out.println(s.match(str,pattern));*/
        /*22.
        Scanner in=new Scanner(System.in);
        String input=in.next();
        String[] inputs = input.split(",") ;
        int leng=inputs.length;
        ListNode [] nodes=new ListNode[leng];
        for(int i=0;i<leng;i++){
            nodes[i]=new ListNode(Integer.parseInt(inputs[i]));
        }
        for(int i=0;i<leng-1;i++){
            nodes[i].next=nodes[i+1];
        }
        ListNode renode=deleteDuplication(nodes[0]);
        while(renode!=null){
            System.out.print(renode.val+" ");
            renode=renode.next;
        } */
        /*23.
        Scanner in=new Scanner(System.in);
        String input=in.next();
        char[] inputs=input.toCharArray();
        System.out.println(isNumeric(inputs));  */
        /*36.
        TreeNode[]nodes=new TreeNode[7];
        for (int i =0;i<7;i++){
            nodes[i]=new TreeNode(i+1);
        }
        nodes[0].left=nodes[1];nodes[0].right=nodes[2];
        nodes[1].left=nodes[3];nodes[1].right=nodes[4];
        nodes[2].left=nodes[5];nodes[2].right=nodes[6];
        ArrayList<Integer> ret=PrintFromTopToBottom(nodes[0]);
        for(int i :ret){
            System.out.print(i+" ");
        }*/
        /*40.
        TreeNode[]nodes=new TreeNode[5];
        int nums[]={10,5,12,4,7};
        for (int i =0;i<5;i++){
            nodes[i]=new TreeNode(nums[i]);
        }
        nodes[0].left=nodes[1];nodes[0].right=nodes[2];
        nodes[1].left=nodes[3];nodes[1].right=nodes[4];
        int target=22;
        Solution s=new Solution();
        ArrayList<ArrayList<Integer>>ret=s.FindPath(nodes[0],target);
        System.out.println("print:");
        for(ArrayList<Integer> i : ret){
            for(int j : i){
                System.out.print(j+" ");
            }
            System.out.println();
        } */
        /*43.
        TreeNode[]nodes=new TreeNode[5];
        int nums[]={10,5,12,4,7};
        for (int i =0;i<5;i++){
            nodes[i]=new TreeNode(nums[i]);
        }
        nodes[0].left=nodes[1];nodes[0].right=nodes[2];
        nodes[1].left=nodes[3];nodes[1].right=nodes[4];
        Solution s=new Solution();
        String str=s.Serialize(nodes[0]);
        //10,5,4,#,#,7,#,#,12,#,#
        System.out.println(str);*/
        /*47.
        Solution s=new Solution();
        Scanner in=new Scanner(System.in);
        String input=in.next();
        char[] inputs=input.toCharArray();
        String caseout="";
        for(char ch : inputs){
            s.Insert(ch);
            caseout = caseout+"-"+s.FirstAppearingOnce();
        }
        System.out.println(caseout);*/
        /*63.
        int[]nums={1,3,4,5,1,5,6,6};
        int[] num1={0};
        int[] num2={0};
        FindNumsAppearOnce(nums,num1,num2);
        System.out.println(num1[0]+"---"+num2[0]);*/
        /*66.
        String s="student. a am I";
        javaCodes j=new javaCodes();
        System.out.println(j.ReverseSentence(s));*/
        /*69.
        javaCodes j=new javaCodes();
        j.dicesSum(2); */
        /*70.
        int[]nums={1,2,4,3,5,0};
        javaCodes j=new javaCodes();
        System.out.println(j.isContinuous(nums)); */
        /*74.
        int[]nums={1,2,4,3,5};
        javaCodes j=new javaCodes();
        j.multiply(nums); */
        /*75.
        String str="-23";
        javaCodes j=new javaCodes();
        System.out.println(j.StrToInt(str)); */

    }
}

class Solution{
    /*14.
    矩阵中路径
    DFS
    private final static int[][] next = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    private int rows;
    private int cols;
    public boolean hasPath(char[] array, int rows, int cols, char[] str) {
        if (rows == 0 || cols == 0)
            return false;
        this.rows = rows;
        this.cols = cols;
        boolean[][] marked = new boolean[rows][cols];
        char[][] matrix = buildMatrix(array);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if (backtracking(matrix, str, marked, 0, i, j))//先比较标记后回溯
                    return true;
        return false;
    }
    private boolean backtracking(char[][] matrix, char[] str, boolean[][] marked, int pathLen, int r,int c) {
        if (pathLen == str.length)//已找到字符串长度的一条路径
            return true;
        if (r < 0 || r >= rows || c < 0 || c >= cols || matrix[r][c] != str[pathLen] || marked[r][c])
            return false;
        marked[r][c] = true;//matrix[r][c] == str[pathLen],将当前位置标记走过
        for (int[] n : next)//判断当前位置的上下左右
            if (backtracking(matrix, str, marked, pathLen + 1, r + n[0], c + n[1]))
                return true;//回溯到当前位置
        marked[r][c] = false;//不符合条件则将当前位置取消标记
        return false;
    }
    private char[][] buildMatrix(char[] array) {//根据数组构建矩阵
        char[][] matrix = new char[rows][cols];
        for (int i = 0, idx = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] = array[idx++];
        return matrix;
    }*/
    /*15.
    机器人的运动范围
    地上有一个 m 行和 n 列的方格。一个机器人从坐标 (0, 0) 的格子开始移动，每一次只能向左右上下四个方向移动一格，
    但是不能进入行坐标和列坐标的数位之和大于 k 的格子。例如，当 k 为 18 时，机器人能够进入方格 (35,37)，因为 3+5+3+7=18。
    但是，它不能进入方格 (35,38)，因为3+5+3+8=19。请问该机器人能够达到多少个格子？
    DFS
    private static final int[][] next = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    private int cnt = 0;
    private int rows;
    private int cols;
    private int threshold;
    private int[][] digitSum;
    public int movingCount(int threshold, int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.threshold = threshold;
        initDigitSum();
        boolean[][] marked = new boolean[rows][cols];
        dfs(marked, 0, 0);
        return cnt;
    }
    private void dfs(boolean[][] marked, int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols || marked[r][c])
            return;
        marked[r][c] = true;//到达当前位置
        if (this.digitSum[r][c] > this.threshold)//超过限定条件
            return;
        cnt++;//当前位置的上下左右
        for (int[] n : next)
            dfs(marked, r + n[0], c + n[1]);
    }
    private void initDigitSum() {//生成各位置的行列数位和
        int[] digitSumOne = new int[Math.max(rows, cols)];
        for (int i = 0; i < digitSumOne.length; i++) {//计算（十位数+个位数+...）
            int n = i;
            while (n > 0) {
                digitSumOne[i] += n % 10;
                n /= 10;//个位数则结束循环，十位...数则也加进来
            }
        }
        this.digitSum = new int[rows][cols];
        for (int i = 0; i < this.rows; i++)//计算（行数位和+列数位和）
            for (int j = 0; j < this.cols; j++)
                this.digitSum[i][j] = digitSumOne[i] + digitSumOne[j];
    }
    */
    /*19.
     打印从 1 到最大的 n 位数
     输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数即999。
     由于 n 位数可能会非常大，因此不能直接用 int 表示数字，而是用 char 数组进行存储。
     回溯法

    public void print1ToMaxOfNDigits(int n) {
        if (n <= 0)
            return;
        char[] number = new char[n];
        print1ToMaxOfNDigits(number, 0);
    }
    private void print1ToMaxOfNDigits(char[] number, int digit) {
        if (digit == number.length) {
            printNumber(number);
            return;
        }
        for (int i = 0; i < 10; i++) {
            number[digit] = (char) (i + '0');
            print1ToMaxOfNDigits(number, digit + 1);
        }
    }
    private void printNumber(char[] number) {
        int index = 0;
        while (index < number.length && number[index] == '0')
            index++;
        while (index < number.length)
            System.out.print(number[index++]);
        System.out.println();
    }

//    过程如下：
//    开始digit=0，i=0,number[0]=0,digit+1
//    digit'=1,i=0,number[1]=0,digit'+1
//    digit''=2,printNumber打印空,return
//    digit'=1,i=1,number[1]=1,digit'+1
//    digit''=2,printNumber打印01,return
//    digit'=1,i=2,number[1]=2,digit'+1
//    digit''=2,printNumber打印02,return
//    ...
//    digit'=1,i=9,number[1]=9,digit'+1
//    digit''=2,printNumber打印09,return
//
//    digit=0，i=1,number[0]=1,digit+1
//    digit'=1,i=0,number[1]=0,digit'+1
//    digit''=2,printNumber打印10,return
//    digit'=1,i=1,number[1]=1,digit'+1
//    digit''=2,printNumber打印11,return
//    digit'=1,i=2,number[1]=2,digit'+1
//    digit''=2,printNumber打印12,return
//    ...
//    digit'=1,i=9,number[1]=9,digit'+1
//    digit''=2,printNumber打印19,return
//
//    ...
*/
    /*20.
    正则表达式匹配。
    请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
    在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配。
    public boolean matchStr(char[] str, int i, char[] pattern, int j) {
        if (i == str.length && j == pattern.length) { // str和pattern都为空
            return true;
        } else if (j == pattern.length) { // pattern为空
            return false;
        }

        boolean flag = false;
        boolean next = (j + 1 < pattern.length && pattern[j + 1] == '*');
        if (next) {// pattern下一个字符是'*'
            if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) { // pattern当前字符匹配，下一字符为*
                return matchStr(str, i, pattern, j + 2) || matchStr(str, i + 1, pattern, j)||matchStr(str, i + 1, pattern, j+2);//当前字符出现1次或多次
            } else {// pattern当前字符无法匹配，下一字符为*，则用下下个字符来跟str匹配比较
                return matchStr(str, i, pattern, j + 2);//当前字符出现0次
            }
        } else {// pattern下一个字符不是'*'
            if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) {// pattern当前字符匹配，下一字符不是*
                return matchStr(str, i + 1, pattern, j + 1);
            } else {// pattern当前字符不匹配，下一字符不是*
                return false;
            }
        }
    }

    public boolean match(char[] str, char[] pattern) {
        return matchStr(str, 0, pattern, 0);
    }
//    当模式中的第二个字符不是“*”时：
//        如果字符串第一个字符和模式中的第一个字符相匹配，那么字符串和模式都后移一个字符，然后匹配剩余的。
//        如果字符串第一个字符和模式中的第一个字符相不匹配，直接返回false。
//    当模式中的第二个字符是“*”时：
//        如果字符串第一个字符跟模式第一个字符不匹配，则模式后移2个字符，继续匹配。
//        如果字符串第一个字符跟模式第一个字符匹配，可以有3种匹配方式：
//            模式后移2字符，相当于x*被忽略；
//            字符串后移1字符，模式后移2字符，相当于x*算一次。
//            符串后移1字符，模式不变，即继续匹配字符下一位，因为*可以匹配多位。
*/
    /*40.
    二叉树中和为某一值的路径
    输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
    private ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
    public ArrayList<ArrayList<Integer>> FindPath(javaCodes.TreeNode root, int target) {
        backtracking(root, target, new ArrayList<>());
        return ret;
    }
    private void backtracking(javaCodes.TreeNode node, int target, ArrayList<Integer> path) {
        if (node == null)
            return;
        path.add(node.val);
        target -= node.val;
        if (target == 0 && node.left == null && node.right == null) {//找到一条路径
            for(int i : path)
                System.out.print(i+" ");
            System.out.println();
            ret.add(new ArrayList<>(path));//path受后面操作影响，因此要另辟一块空间保存path此时的值
        } else {//在子树中找
            backtracking(node.left, target, path);
            backtracking(node.right, target, path);
        }
        path.remove(path.size() - 1);//递归结束后删掉本轮递归添加的元素
    }
     */
    /*43.
    序列化二叉树
    请实现两个函数，分别用来序列化和反序列化二叉树
    二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。
    序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过某种符号表示空节点（#），
    以 ！ 表示一个结点值的结束（value!）。
    二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。
    private String deserializeStr;
    public String Serialize(javaCodes.TreeNode root) {//先序
        if (root == null)
            return "#";
        return root.val + "," + Serialize(root.left) + "," + Serialize(root.right);
    }//递归
    public javaCodes.TreeNode Deserialize(String str) {
        deserializeStr = str;
        return Deserialize();
    }
    private javaCodes.TreeNode Deserialize() {
        if (deserializeStr.length() == 0)
            return null;
        int index = deserializeStr.indexOf(",");//先序取根节点
        String node = index == -1 ? deserializeStr : deserializeStr.substring(0, index);
        deserializeStr = index == -1 ? "" : deserializeStr.substring(index + 1);//修改deserializeStr以便后续遍历
        if (node.equals("#"))//返回
            return null;
        int val = Integer.valueOf(node);
        javaCodes.TreeNode t = new javaCodes.TreeNode(val);
        t.left = Deserialize();//左右子树分别根据deserializeStr去递归调用
        t.right = Deserialize();
        return t;
    }
    //另一种稍好理解的写法：
    //使用index来设置树节点的val值，递归遍历左节点和右节点，如果值是#则表示是空节点，返回
    int index=-1;
    javaCodes.TreeNode Deserialize(String str) {
        String[] s = str.split(",");//将序列化之后的序列用，分隔符转化为数组
        index++;
        int len = s.length;
        if (index > len) {
            return null;
        }
        javaCodes.TreeNode treeNode = null;
        if(s[index].equals("#"))
            return null;
        else{//不是叶子节点，递归
            treeNode = new javaCodes.TreeNode(Integer.parseInt(s[index]));
            treeNode.left = Deserialize(str);
            treeNode.right = Deserialize(str);
        }
        return treeNode;
    }
    */

}

