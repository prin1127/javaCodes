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


}

