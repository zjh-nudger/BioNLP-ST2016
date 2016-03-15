package pro;

import java.io.File;
import java.util.ArrayList;

import util.AllRoot;
import util.WFileWriter;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//test1
		/*训练集*/
//		File a1 = new File(AllRoot.trainPath + "a1");
//		File gdep = new File(AllRoot.trainPath + "gdep_process");
//		File a2 = new File(AllRoot.trainPath + "a2");
		/*发展集*/
//		File a1 = new File(AllRoot.devPath + "a1");
//		File gdep = new File(AllRoot.devPath + "gdep_process");
//		File a2 = new File(AllRoot.devPath + "a2");
		/*得到实例*/
//		File a1s[] = a1.listFiles();
//		File gdeps[] = gdep.listFiles();
//		File a2s[] = a2.listFiles();
//		
//		ArrayList<String> allSample = new ArrayList<String>();
//		for (int i = 0; i < a1s.length; i++) {
//			Sample.setSample(a1s[i], gdeps[i], a2s[i], allSample);
//		}
//		WFileWriter.writeArrayListToFile(allSample, new File("F:/train_Li_distance")); //所有实例
		
		//test2
		//得到预测答案文档a2(此过程会过滤掉不符合元素类型规则的实例)
//		File predicFile = new File(AllRoot.devPath + "li-out-3.csv");
//		File sampleFile = new File(AllRoot.devPath + "a2_2");
//		Postpro.getA2Answer(predicFile, sampleFile);
		
		//test3
		/*评价*/
//		File predictFile = new File(AllRoot.devPath + "a2_predict");
//		File a2File = new File(AllRoot.devPath + "a2");
//		Evaluate.evaluation(predictFile, a2File);
		
		File trainFile = new File("F:/train_Li_distance");
		File devFile = new File("F:/dev_Li_distance");
		File testFile = new File("F:/dev_Li_distance");
		SVM.setSample(trainFile, devFile, testFile);
	}

}
