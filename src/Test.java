import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import javax.tools.JavaFileManager.Location;

import process.Statistics;
import util.AllRoot;
import util.WFileReader;
import util.WFileWriter;


public class Test {

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		//test1 统计a2文件
//		File file = new File(AllRoot.trainPath+ "a2");
//		String path = AllRoot.trainPath;
////		File file = new File(AllRoot.devPath+ "a2");
////		String path = AllRoot.devPath;
//		
//		Statistics.toOneFile(file, path);
		
		
		//test3  输出每个单词的偏移量
//		File txtFile = new File(AllRoot.trainPath+ "txt");
//		File gtFile = new File(AllRoot.trainPath+ "gt");
		
		File txtFile = new File(AllRoot.devPath+ "txt");
		File gtFile = new File(AllRoot.devPath+ "gt");
		
		process.Location.setPosition(txtFile, gtFile, 0, true);
	}

}
class SortByIndex implements Comparator<Object> {
	public int compare(Object o1, Object o2) {
		String s1 = (String) o1;
		String s2 = (String) o2;
		Integer temp1 = Integer.valueOf(s1.split(" ")[0]);
		Integer temp2 = Integer.valueOf(s2.split(" ")[0]);
		return temp1.compareTo(temp2);
	}	
}
