package pro;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

import util.WFileReader;
import util.WFileWriter;

public class SVM {

	public static void setSample(File trainFile, File devFile, File testFile) {
		
		ArrayList<String> trainList = WFileReader.getFileLine(trainFile);
		ArrayList<String> devList = WFileReader.getFileLine(devFile);
		//ArrayList<String> testList = WFileReader.getFileLine(testFile);
		
		ArrayList<String> outTrainList = new ArrayList<>();
		ArrayList<String> outDevList = new ArrayList<>();
		
		HashMap<String, Integer> vocabularyMap = new HashMap<>();
		int count = 0;
		
		for(String s : trainList) {
			String ss[] = s.split(" ");
			StringBuffer sb = new StringBuffer();
			if(ss[0].equals("0")) sb.append("23" + " ");
			else sb.append(ss[0] + " ");
			for (int i = 1; i < ss.length; i++) {
				if (vocabularyMap.containsKey(ss[i])) {
					int value = vocabularyMap.get(ss[i]);
					if(!sb.toString().contains(value + ":1 ")) sb.append(value + ":1 ");
				}else {
					vocabularyMap.put(ss[i], ++count);
					if(!sb.toString().contains(count + ":1 ")) sb.append(count + ":1 ");
				}
			}
			outTrainList.add(featureSort(sb.toString()));
		}
		System.out.println(vocabularyMap.size());
		int dev = 0;
		for(String s : devList) {
			String ss[] = s.split(" ");
			StringBuffer sb = new StringBuffer();
			if(ss[0].equals("0")) sb.append("23" + " ");
			else {
				++dev;
				sb.append(ss[0] + " ");
			}
			for (int i = 1; i < ss.length; i++) {
				if (vocabularyMap.containsKey(ss[i])) {
					int value = vocabularyMap.get(ss[i]);
					if(!sb.toString().contains(value + ":1 ")) sb.append(value + ":1 ");
				}else {
					if(!sb.toString().contains(String.valueOf((vocabularyMap.size()+1) + ":1 "))) sb.append((vocabularyMap.size()+1) + ":1 ");
				}
			}
			outDevList.add(featureSort(sb.toString()));
		}
		System.out.println(dev);
		WFileWriter.writeArrayListToFile(outTrainList, new File("F:/train_svm"));
		WFileWriter.writeArrayListToFile(outDevList, new File("F:/dev_svm"));
	}
	
	public static String featureSort(String s) {

		ArrayList<Integer> tempList = new ArrayList<>();
		String ss[] = s.split(" ");
		for (int i = 1; i < ss.length; i++) {
			tempList.add(Integer.valueOf(ss[i].split(":")[0]));
		}
		tempList = util.ListSort.listSort(tempList);
		StringBuffer sb = new StringBuffer();
		sb.append(ss[0] + " ");
		for (int i = 0; i < tempList.size(); i++) {
			sb.append(tempList.get(i) + ":1"+ " ");
		}
		return sb.toString().substring(0, sb.toString().length()-1);
	}
}
