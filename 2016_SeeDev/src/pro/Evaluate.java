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
		for (int i = 0; i < pFiles.length; i++) {
			ArrayList<String> pList = WFileReader.getFileLine(pFiles[i]);
			tp_fp += pList.size();
			HashSet<String> a2Hash = WFileReader.getFileLine_HashSet2(a2Files[i]);
			for (int j = 0; j < pList.size(); j++) {
				String onePredict = pList.get(j).split("\t")[1];
				if(a2Hash.contains(onePredict)) ++tp;
			}
		}
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
	}
}
