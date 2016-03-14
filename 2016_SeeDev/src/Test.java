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
//		File txtFile = new File(AllRoot.trainPath+ "txt");
//		File gtFile = new File(AllRoot.trainPath+ "gt");
//		File stanfordFile = new File(AllRoot.trainPath+ "stanford");
		
//		File txtFile = new File(AllRoot.devPath+ "txt");
//		File gtFile = new File(AllRoot.devPath+ "gt");
//		File stanfordFile = new File(AllRoot.devPath+ "stanford");
		
//		process.Location.setPosition(txtFile, gtFile, 0, true);
//		process.Location.setPosition(txtFile, stanfordFile, 1, false);

//		File txtFile = new File(AllRoot.trainPath+ "txt");
//		File gdepFile = new File(AllRoot.trainPath+ "gdep");
		
//		File txtFile = new File(AllRoot.devPath+ "txt");
//		File gdepFile = new File(AllRoot.devPath+ "gdep");
//		
//		process.Location.setPosition(txtFile, gdepFile, 1, true);
		
		//整理顺序
//		File temp = new File("K:/dev_self");
//		ArrayList<String> tempList = WFileReader.getFileLine(temp);
//		ArrayList<String> newList = new ArrayList<>();
//		for (String s : tempList) {
//			String ss[] = s.split(" ");
//			StringBuffer sb = new StringBuffer();
//			sb.append(ss[0] + " ");
//			for (int i = 3; i < ss.length; i++) {
//				sb.append(ss[i] + " ");
//			}			
//			s = sb.toString() + ss[1] + " "+ ss[2];
//			newList.add(s);
//		}
//		WFileWriter.writeArrayListToFile(newList, new File("K:/dev_self_new"));
		
		//test 单个类别
//		File temp = new File("K:train_entity_type");
//		ArrayList<String> tempList = WFileReader.getFileLine(temp);
//		ArrayList<String> newList = new ArrayList<>();
//		ArrayList<String> allList = new ArrayList<>();
//		for (String s : tempList) {
//			if(!s.startsWith("0")){
//				allList.add(s);
//				String ss[] = s.split(" ");
//				if(ss[1].equals(ss[2])) System.out.println(s);
//			}
//			//System.out.println(s);
//			
//			if(s.startsWith("21")){ s = s.replaceFirst("21", "1"); newList.add(s);continue;} 
//			if(s.startsWith("14")){ s = s.replaceFirst("14", "2"); newList.add(s);continue;}
//			if(s.startsWith("22")){ s = s.replaceFirst("22", "3"); newList.add(s);continue;}
//			if(s.startsWith("10")){ s = s.replaceFirst("10", "4"); newList.add(s);continue;}
//			if(s.startsWith("8")){  s = s.replaceFirst("8", "5"); newList.add(s);continue;}
//			if(s.startsWith("12")){ s = s.replaceFirst("12", "6"); newList.add(s);continue;}
//			if(s.startsWith("17")){ s = s.replaceFirst("17", "7"); newList.add(s);continue;}
//			if(s.startsWith("6")){  s = s.replaceFirst("6", "8"); newList.add(s);continue;}
//			if(s.startsWith("3")){  s = s.replaceFirst("3", "9"); newList.add(s);continue;}
//			if(s.startsWith("0")) {
//				String ss[] = s.split(" ");
//				if(s.contains("Regulatory_Network")) {newList.add(s);continue;}
//				if(s.contains("Pathway")) {newList.add(s);continue;}
//				if(s.contains("Genotype")) {newList.add(s);continue;}
//				if(s.contains("Tissue")) {newList.add(s);continue;}
//				if(s.contains("Development_phase")) {newList.add(s);continue;}
//			}
//		} //for
//		WFileWriter.writeArrayListToFile(newList, new File("K:/train_9c"));
//		Collections.sort(allList, new SortByIndex());
//		WFileWriter.writeArrayListToFile(allList, new File("K:/all_train"));
		
		File temp = new File("K:/dev_entity_type");
		ArrayList<String> tempList = WFileReader.getFileLine(temp);
		ArrayList<String> newList = new ArrayList<>();
		for (String s : tempList) {
			if(s.startsWith("1 ")){
				newList.add(s);
			} else if (s.startsWith("0")) {
				if(s.contains("Protein_Family Protein")) {newList.add(s); continue;}
				if(s.contains("Protein Protein_Family")) {newList.add(s); continue;}
				if(s.contains("Protein_Family Protein_Family")) {newList.add(s); continue;}
				if(s.contains("Gene Gene_Family")) {newList.add(s); continue;}
				if(s.contains("Gene_Family Gene")) {newList.add(s); continue;}
				if(s.contains("Gene_Family Gene_Family")) {newList.add(s); continue;}
				if(s.contains("RNA RNA")) {newList.add(s); continue;}
			}
		}
		WFileWriter.writeArrayListToFile(newList, new File("K:/dev1"));
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
