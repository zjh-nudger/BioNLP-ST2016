package process;

import java.io.File;
import java.util.ArrayList;

import util.WFileReader;
import util.WFileWriter;

public class Location {

	public static void setPosition(File txtFile, File supportFile, int col, boolean flag) {
		
		File txtFiles[] = txtFile.listFiles();
		File supportFiles[] = supportFile.listFiles();
		int docNum = txtFiles.length;
		
		for (int i = 0; i < docNum; i++) {
			
			File oneTxt = txtFiles[i];
			File oneSupport = supportFiles[i];
			
			String txtString = WFileReader.getFileToOneLine(oneTxt).toString();
			ArrayList<String> newList = new ArrayList<>();
			ArrayList<String> supportLineList = WFileReader.getFileLine(oneSupport);

			int offset = 0;
			for (int j = 0; j < supportLineList.size(); j++) {
				if(supportLineList.get(j).length() == 0) {
					newList.add(supportLineList.get(j));
					j++;
					continue;
				}

				String word = supportLineList.get(j).split("\t")[col];
				int wordStart = txtString.indexOf(word, offset);
				int wordEnd = wordStart + word.length();
				offset = wordEnd;
				String newLine = supportLineList.get(j) + "\t" + wordStart + "\t" + wordEnd;
				newList.add(newLine);
			}
			File outFile;
			if(flag) outFile = new File(oneSupport.getAbsolutePath().replace("\\gt\\", "\\gt_process\\"));
			else outFile = new File(oneSupport.getAbsolutePath().replace("\\stanford\\", "\\stanford_process\\"));
			WFileWriter.writeArrayListToFile(newList, outFile);
		}
		
	}
}
