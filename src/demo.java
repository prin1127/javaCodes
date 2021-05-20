import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;

class demo {
    public static int fx(int n) {
        return n;
    }
    public static void main(String[]args) throws IOException {
        BufferedReader reader=new BufferedReader(new InputStreamReader(System.in));
        int num=Integer.parseInt(reader.readLine());
        System.out.println(fx(num));
    }
}
/*
笔试题：
0.（测试）整数拆分问题
可参考：https://www.cnblogs.com/vectors07/p/8034636.html
1.并行 分片 均匀
可参考：https://blog.csdn.net/zhaohansk/article/details/77411552
2.Map和LinkedList实现kv缓存最久未使用算法更新
可参考：https://www.cnblogs.com/yjxyy/p/11098732.html
3.数字串珠，要求1|3|5|6|9为偶数个，求最长长度

 */