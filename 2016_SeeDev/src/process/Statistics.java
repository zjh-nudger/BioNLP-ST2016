package process;

import java.io.File;
import java.util.ArrayList;

import util.EventClass;
import util.WFileReader;
import util.WFileWriter;

public class Statistics {

	public static void toOneFile(File file, File outFile) {
		//将a2文档归到一个文档中，统计各个类别关系数目
		File[] files = file.listFiles();
		ArrayList<String> allLineList = new ArrayList<>();
		EventClass eClass = new EventClass();
		for (int i = 0; i < eClass.getSize(); i++) {
			int count = 0;
			String className = eClass.getClassName(i);
			//System.out.println("%" + className);
			for (int j = 0; j < files.length; j++) {
				
				ArrayList<String> oneDocList = WFileReader.getFileLine(files[j]);
				for(String s : oneDocList){
					//System.out.println("#" + s.split(" ")[0].split("\t")[1]);
					if (className.equals(s.split(" ")[0].split("\t")[1])) {
						count++;
						allLineList.add(s + "\t\t" + files[j].getName());
					}
				}
			}			
			System.out.println(className + ":" + count);
		}
		WFileWriter.writeArrayListToFile(allLineList, outFile);
		
	}
	
	public static void txtToOneLine(File file) {
		
		File[] files = file.listFiles();
		for (int i = 0; i < files.length; i++) {
			String pathName = files[i].getAbsolutePath();
			StringBuffer sb = WFileReader. getFileToOneLine(files[i]);
			WFileWriter.writeStringToFile(sb.toString(), new File(pathName.replace("\\txt\\", "\\txt_process\\")));
		}
	}
}
