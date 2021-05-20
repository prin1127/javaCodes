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