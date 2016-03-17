package util;

import java.io.File;
import java.util.ArrayList;

import util.AllRoot;
import util.WFileReader;

public class EventClass {

	static ArrayList<String> EClass = new ArrayList<String>();
	
	public EventClass() {
		File file = new File(AllRoot.trainPath + "EventClass.txt");
		ArrayList<String> lines = WFileReader.getFileLine(file);
		EClass.add("Not");
		for(String line : lines) {
			EClass.add(line);
		}
	}
	
	public int getClassIndex(String str) {		
		return getEClass().indexOf(str);
	}
	
	public String getClassName(int i) {		
		return getEClass().get(i);
	}

	public static ArrayList<String> getEClass() {
		return EClass;
	}

	public int getSize() {
		return EClass.size();
	}
}
