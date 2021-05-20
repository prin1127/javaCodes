import java.io.*;

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