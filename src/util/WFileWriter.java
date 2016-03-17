package util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;

public class WFileWriter {
	
	public static void writeStringToFile(String string, File file){		
		//将候选trigger写入文件
		try {			
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			bw.write(string);
			bw.close();			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void writeHashSetToFile(HashSet set, File file){		
		//将候选trigger写入文件
		try {			
			BufferedWriter TriCanBw = new BufferedWriter(new FileWriter(file));
			Iterator<String> iterator = set.iterator();
			while(iterator.hasNext()){
				TriCanBw.append(iterator.next() + "\r\n");
			}
			TriCanBw.close();			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void writeHashMapToFile(HashMap set, File file){		
		//将候选trigger写入文件
		try {			
			BufferedWriter TriCanBw = new BufferedWriter(new FileWriter(file));
			Iterator iterator = set.entrySet().iterator();
			while(iterator.hasNext()){
				Entry e = (Entry) iterator.next();
				TriCanBw.append( e.getKey() +" "+e.getValue()+ "\r\n");
			}
			TriCanBw.close();			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void writeArrayListToFile(ArrayList set, File file){		
		//将候选trigger写入文件
		try {			
			BufferedWriter TriCanBw = new BufferedWriter(new FileWriter(file));
			for(Object word: set){
				TriCanBw.write(word.toString());
				TriCanBw.newLine();
				TriCanBw.flush();
			}			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
