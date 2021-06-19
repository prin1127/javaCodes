/*public class DoubleLinkTest<T> {

    //内部构造节点类
    private class Node<T> {
        private T data;
        private Node next; // 指向下一个节点的引用
        private Node prev; // 指向前一个节点的引用

        public Node(T data) {
            this.data = data;
        }
    }

    private Node<T> head; // 模拟头结点
    private Node<T> last; // 模拟尾部节点
    private Node<T> tmp; // 暂定一个临时节点，用作指针节点
    private int length;

    public void DoubleLinkTest() {
        head = new Node<T>(null);
        last = head;
        length = 0;
    }

    public void DoubleLinkTest(T data) {
        head = new Node<T>(data);
        last = head;
        length = 0;
    }

    //判断链表是否为空
    public boolean isEmpty() {
        return length == 0;
    }

    //普通添加，往链表尾部添加
    public void add(T data) {
        if (isEmpty()) { // 链表为空，新创建一个链表
            head = new Node<T>(data);
            last = head;
            length++;
        } else {
            tmp = new Node<T>(data);
            tmp.prev = last;
            last.next = tmp; // 将新的节点与原来的尾部节点进行结构上的关联
            last = tmp; // tmp将成为最后一个节点
            length++;
        }
    }

    //在data后面添加数据insertData
    public void addAfter(T data, T insertData) {
        tmp = head;
        while (tmp != null) { // 我们假定这个head是不为空的。
            if (tmp.data.equals(data)) {
                Node<T> tmpNew = new Node<T>(insertData);
                tmpNew.prev = tmp;
                tmpNew.next = tmp.next;// 对新插入的数据进行一个指向的定义
                tmp.next.prev=tmpNew;
                tmp.next = tmpNew;

                if (tmpNew.next == null) {
                    last = tmpNew;
                }
                length++;
            }
            tmp = tmp.next;
        }
    }

    //删除，删除指定的数据
    public void remove(T data) {
        tmp = head;// 我们假定这个head是不为空的。
        while (tmp != null) {
            if (tmp.data.equals(data)) {
                tmp.prev.next = tmp.next;
                tmp.next.prev=tmp.prev;
                length--;
            }
            tmp = tmp.next;
        }

    }

    *//**
     * 测试打印数据
     *//*
    public void printList() {
        tmp = head;
        for (int i = 0; i < length-1; i++) {
            System.out.print(tmp.data + "  ");
            tmp = tmp.next;
        }
        System.out.println(tmp.data);
    }

    public void printListReverse() {
        tmp = last;
        for (int i = 0; i < length-1; i++) {
            System.out.print(tmp.data + "  ");
            tmp = tmp.prev;
        }
        System.out.println(tmp.data);
    }

    public static void main(String[] args) {

        DoubleLinkTest<Integer> link = new DoubleLinkTest<Integer>();
        link.add(1);
        link.add(2);
        link.add(3);
        link.add(5);
        link.add(6);
        link.add(7);
        link.printList();
        link.printListReverse();

        System.out.println(" ============== ");

        System.out.println(" ==== 在3后面添加一个数据4========== ");
        link.addAfter(3, 4);
        link.printList();
        link.printListReverse();

        System.out.println(" ==== 移除一个数据4开始========== ");
        link.remove(4);
        link.printList();
        link.printListReverse();

    }

}*/

public class DoubleLinkTest {

    private class Node {
        private long data;
        private Node next; // 指向下一个节点的引用
        private Node previous; // 指向前一个节点的引用

        public Node(long data) {
            this.data = data;
        }
    }

    private Node first;//头
    private Node last;//尾


    //从头部插入数据
    public void insertFirst(long value){
        Node newNode = new Node(value);
        if (first == null) {
            last = newNode;
        }else {
            first.previous = newNode;
            newNode.next = first;
        }
        first = newNode;
    }

    //从尾部插入数据
    public void insertLast(long value){
        Node newNode = new Node(value);
        if (first == null) {
            first = newNode;
        }else {
            last.next = newNode;
            newNode.previous = last;
        }
        last = newNode;
    }

    public boolean isEmpty(){
        return first == null;
    }

    //删除头节点
    public Node deleteFirst(){
        if (first == null) {
            throw new RuntimeException("链表数据不存在");
        }
        Node temp = first;
        if (first.next == null) {
            last = null;
        }else {
            first.next.previous = null;
        }
        first = temp.next;
        return temp;
    }

    //删除尾节点
    public Node deleteLast(){
        if (first == null) {
            throw new RuntimeException("链表数据不存在");
        }
        Node temp = last;
        if (first.next == null) {//把第一个删除
            last = null;
            first = null;
        }else {
            last.previous.next = null;
        }
        last = temp.previous;
        return temp;
    }

    //删除指定值节点
    public Node deleteByKey(long key){
        Node current = first;
        while(current.data != key){
            if (current.next == null) {
                System.out.println("没找到节点");
                return null;
            }
            current = current.next;
        }
        if (current == first) {
            return deleteFirst();
        }else if(current==last){
            return deleteLast();
        }else{
            current.previous.next = current.next;
            current.next.previous = current.previous;
        }
        return current;
    }


    public void printList() {
        Node tmp = first;
        while(tmp!=null) {
            System.out.print(tmp.data + "  ");
            tmp = tmp.next;
        }
        System.out.println();
    }

    public void printListReverse() {
        Node tmp = last;
        while(tmp!=null) {
            System.out.print(tmp.data + "  ");
            tmp = tmp.previous;
        }
        System.out.println();
    }


    //查找指定值节点
    public Node findByKey(long key) {
        Node current = first;
        while (current.data != key) {
            if (current.next == null) {
                System.out.println("没找到");
                return null;
            }
            current = current.next;
        }
        return current;
    }

    //根据索引查找对应的值
    public Node findByPosition(int position){
        Node current = first;
        //为什么是position - 1，因为要使用遍历，让current指向下一个， 所以position - 1的下个node就是要找的值
        for (int i = 0; i < position - 1 ; i++) {
            current  = current.next;
        }
        return current;
    }


    public static void main(String[] args) {
        DoubleLinkTest linkList = new DoubleLinkTest();

        linkList.insertFirst(21);
        linkList.insertFirst(22);
        linkList.insertFirst(23);
        System.out.println("-从头插入---------------------------------------");
        linkList.printList();
        linkList.printListReverse();
        linkList.insertLast(24);
        linkList.insertLast(25);
        linkList.insertLast(26);
        linkList.insertLast(27);
        System.out.println("-从尾插入---------------------------------------");
        linkList.printList();
        linkList.printListReverse();
        System.out.println("-删除指定值27---------------------------------------");
        linkList.deleteByKey(27);
        linkList.printList();
        linkList.printListReverse();
        System.out.println("-删除指定值21---------------------------------------");
        linkList.deleteByKey(21);
        linkList.printList();
        linkList.printListReverse();


    }
}
