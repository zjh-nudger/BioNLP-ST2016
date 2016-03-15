package pro;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;

import util.WFileReader;

public class Evaluate {

	public static void evaluation(File predictFile, File a2File) {
		
		File pFiles[] = predictFile.listFiles();
		File a2Files[] = a2File.listFiles();
		
		double tp = 0;
		double all = 819; //发展集中的所有正例
		double tp_fp = 0;
		double tp2 = 0;
		double temp = 0;
		for (int i = 0; i < pFiles.length; i++) {
			ArrayList<String> pList = WFileReader.getFileLine(pFiles[i]);
			tp_fp += pList.size();
			HashSet<String> a2Hash = WFileReader.getFileLine_HashSet2(a2Files[i]);
			ArrayList<String> a2List = WFileReader.getFileLine(a2Files[i]);
			for (int j = 0; j < pList.size(); j++) {
				String onePredict = pList.get(j).split("\t")[1];
				if(a2Hash.contains(onePredict)) ++tp;
				else {
					String type = onePredict.split(" ")[0];
					String e1 = onePredict.split(" ")[1].split(":")[1] + " ";
					String e2 = onePredict.split(" ")[2].split(":")[1] + " "; //若不加空格，会造成T1和T11匹配的情况，应注意
					//System.out.println(type + " " + e1 + " " + e2);
					for (int k = 0; k < a2List.size(); k++) {
						String s = a2List.get(k) + " ";
						if (s.contains(type) & s.contains(e1) & s.contains(e2)) {
							++tp2;
							if(onePredict.contains("Element1") & onePredict.contains("Element2")) {tp++; temp++;}
							else {
								System.out.println(a2Files[i].getAbsolutePath());
								System.out.println(onePredict);
							}							
						}
					}
				}
			}
		}
		//tp = tp + (tp2-temp); //模糊匹配
		double fp = tp_fp - tp;
		double recall = tp / all;
		double precision = tp / tp_fp;
		double F = (2*recall*precision) / (recall+precision);
		System.out.println("tp: " + tp);
		System.out.println("fp: " + fp);
		System.out.println("all: " + all);
		System.out.println("recall: " + recall);
		System.out.println("precision: " + precision);
		System.out.println("F: " + F);
		System.out.println("元素分配错误: " + tp2);
		System.out.println("元素分配错误中，含有Element1和2元素的: " + temp);
	}
}
