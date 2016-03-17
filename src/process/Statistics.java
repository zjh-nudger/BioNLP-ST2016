package process;

import java.io.File;
import java.util.ArrayList;

import util.AllRoot;
import util.EventClass;
import util.WFileReader;
import util.WFileWriter;

public class Statistics {

	public static void toOneFile(File file, String path) {
		//统计a2
		File[] files = file.listFiles();
		ArrayList<String> a2lLineList = new ArrayList<>();
		ArrayList<String> equalEntityList = new ArrayList<>();
		for (int i = 0; i < files.length; i++) {
			ArrayList<String> oneA2List = WFileReader.getFileLine(files[i]);
			for(String s : oneA2List) {
				if(s.startsWith("R")) a2lLineList.add(s);
				else equalEntityList.add(s);
			}
		}
		WFileWriter.writeArrayListToFile(a2lLineList, new File(path + "a2_All"));
		WFileWriter.writeArrayListToFile(equalEntityList, new File(path + "equalEntity"));
		
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
