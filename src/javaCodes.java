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

    /*
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

    /*
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
    }
}

