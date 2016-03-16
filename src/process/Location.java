package process;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

import util.WFileReader;
import util.WFileWriter;

public class Location {

	public static void setPosition(File txtFile, File supportFile, int col, boolean flag) {
		
		File txtFiles[] = txtFile.listFiles();
		File supportFiles[] = supportFile.listFiles();
		int docNum = txtFiles.length;
		
		//HashSet<String> temp = new HashSet<>();
		
		for (int i = 0; i < docNum; i++) {
			
			File oneTxt = txtFiles[i];
			File oneSupport = supportFiles[i];
			
			String txtString = WFileReader.getFileToOneLine(oneTxt).toString();
			ArrayList<String> newList = new ArrayList<>();
			ArrayList<String> supportLineList = WFileReader.getFileLine(oneSupport);

			int offset = 0;
			for (int j = 0; j < supportLineList.size(); j++) {
				
				String oneLine = supportLineList.get(j);
				if(oneLine.length() == 0) {
					newList.add(oneLine);
					continue;
				}

				String word = oneLine.split("\t")[col];
				int wordStart = txtString.indexOf(word, offset);
				//if(wordStart == -1) temp.add(word);

				if(wordStart == -1) {
					String tmp = word;
					if(word.equals("''") | word.equals("``")) word = "\"";
					else word = word.replace("-LSB-", "[").replace("-RSB-", "]").replace("-LRB-", "(").replace("-RRB-", ")");
					wordStart = txtString.indexOf(word, offset);
					oneLine = oneLine.replace(tmp, word);
					System.out.println(word);
				}
				
				int wordEnd = wordStart + word.length();
				offset = wordEnd;
				String newLine = oneLine + "\t" + wordStart + "\t" + wordEnd;
				newList.add(newLine);
			}
			File outFile;
			if(flag) {
				outFile = new File(oneSupport.getAbsolutePath().replace("\\gt\\", "\\gt_process\\"));
				//outFile = new File(oneSupport.getAbsolutePath().replace("\\gdep\\", "\\gdep_process\\"));
			}
			else outFile = new File(oneSupport.getAbsolutePath().replace("\\stanford\\", "\\stanford_process\\"));
			WFileWriter.writeArrayListToFile(newList, outFile);
		}
//		Iterator it = temp.iterator();
//		while(it.hasNext()){
//			System.out.println(it.next());
//		}
		
	}
	
}
