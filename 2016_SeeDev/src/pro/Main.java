package pro;

import java.io.File;
import java.util.ArrayList;

import util.AllRoot;
import util.WFileWriter;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
//		File a1 = new File(AllRoot.trainPath + "a1");
//		File gdep = new File(AllRoot.trainPath + "gdep_process");
//		File a2 = new File(AllRoot.trainPath + "a2");
		
//		File a1 = new File(AllRoot.devPath + "a1");
//		File gdep = new File(AllRoot.devPath + "gdep_process");
//		File a2 = new File(AllRoot.devPath + "a2");
//		
//		File a1s[] = a1.listFiles();
//		File gdeps[] = gdep.listFiles();
//		File a2s[] = a2.listFiles();
//		
//		ArrayList<String> allSample = new ArrayList<String>();
//		for (int i = 0; i < a1s.length; i++) {
//			Sample.setSample(a1s[i], gdeps[i], a2s[i], allSample);
//		}
//		WFileWriter.writeArrayListToFile(allSample, new File("F:/dev_Li"));
		
//		File predicFile = new File(AllRoot.devPath + "li-out-3.csv");
//		File sampleFile = new File(AllRoot.devPath + "a2_2");
//		Postpro.getA2Answer(predicFile, sampleFile);
		
		File predictFile = new File(AllRoot.devPath + "a2_predict");
		File a2File = new File(AllRoot.devPath + "a2");
		Evaluate.evaluation(predictFile, a2File);
	}

}
