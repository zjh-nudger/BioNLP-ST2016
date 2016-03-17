package process;

public class test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String word = "``";
		if(word.equals("''") | word.equals("``")) word = "\"";
		word = "-LSB-39-RSB-";
		word = word.replaceAll("-LSB-", "[").replace("-RSB-", "]").replace("-LRB-", "(").replace("-RRB-", ")");
		System.out.println(word);
	}

}
