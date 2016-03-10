import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;

import javax.tools.JavaFileManager.Location;

import process.Statistics;
import util.AllRoot;
import util.WFileReader;


public class Test {

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

//		//test1
//		File file = new File(AllRoot.trainPath+ "a2");
//		File outFile = new File(AllRoot.trainPath + "a2_All");
////		File file = new File(AllRoot.devPath+ "a2");
////		File outFile = new File(AllRoot.devPath + "a2_All");
//		Statistics.toOneFile(file, outFile);
		
		//test2
////		File file = new File(AllRoot.devPath+ "txt");
//		File file = new File(AllRoot.trainPath+ "txt");
//		Statistics.txtToOneLine(file);

//		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File("F:/SeeDev-binary-10662856-1.txt")), "UTF-8"));
//		String s = br.readLine();
//		s = br.readLine();
//		System.out.println(s);
		
		//test3  输出每个单词的偏移量
////		File txtFile = new File(AllRoot.trainPath+ "txt");
////		File gtFile = new File(AllRoot.trainPath+ "gt");
////		File stanfordFile = new File(AllRoot.trainPath+ "stanford");
//		
//		File txtFile = new File(AllRoot.devPath+ "txt");
//		File gtFile = new File(AllRoot.devPath+ "gt");
//		File stanfordFile = new File(AllRoot.devPath+ "stanford");
//		
//		process.Location.setPosition(txtFile, gtFile, 0, true);
//		process.Location.setPosition(txtFile, stanfordFile, 1, false);
		
		//test4
		File a1File = new File("K:/SeeDev-binary-9657152-1.a1");
		File stanfordFile = new File("K:/SeeDev-full-9657152-1.txt.conllx");

	}

}
