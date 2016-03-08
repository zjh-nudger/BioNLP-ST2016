package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;

public class WFileReader {
	private static BufferedReader br;
	private static BufferedReader br2;
	private static BufferedReader br3;

	public static StringBuffer getFileToOneLine(File source){		
		StringBuffer result = new StringBuffer();
		try {
			br2 = new BufferedReader(new InputStreamReader(new FileInputStream(source), "UTF-8"));		
			
			String singLine = "";
			while((singLine = br2.readLine()) != null){			
				result.append(singLine + " ");
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		return result;
	}
	
	public static StringBuffer getFile2(File source){		
		StringBuffer result = new StringBuffer();
		try {
			br2 = new BufferedReader(new FileReader(source));		
			
			String singLine = "";
			while((singLine = br2.readLine()) != null){			
				result.append(singLine + "\n");
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		return result;
	}
	
	public static String getFile(File source){		
		StringBuffer result = new StringBuffer();
		try {
			br2 = new BufferedReader(new FileReader(source));		
			
			String singLine = "";
			while((singLine = br2.readLine()) != null){			
				result.append(singLine + " ");
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		return result.toString();
	}
	
	public static ArrayList<String> getFileLine(File source){		
		ArrayList<String> result = new ArrayList<String>();
		try {
			br2 = new BufferedReader(new FileReader(source));		
			
			String singLine = "";
			while((singLine = br2.readLine()) != null){			
				result.add(singLine);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		return result;
	}
	
	public static ArrayList<String> getFileLine_head(File source){		
		ArrayList<String> result = new ArrayList<String>();
		try {
			br3 = new BufferedReader(new FileReader(source));		
			
			String singLine = "";
			while((singLine = br3.readLine()) != null){			
				result.add(singLine.substring(0,5));	
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		return result;
	}

	public static HashSet<String> getFileLine_HashSet(File file) {
		HashSet<String> result = new HashSet<String>();
		
		try {
			br = new BufferedReader(new FileReader(file));		
			
			String singLine = "";
			while((singLine = br.readLine()) != null){			
				result.add(singLine);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
		return result;
	}	
	
}
