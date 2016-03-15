package pro;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import util.EventClass;
import util.WFileReader;
import util.WFileWriter;

public class Sample {

	public static void setSample(File a1File, File gdepFile, File a2File, ArrayList<String> allSample) {
		
		Pattern p = Pattern.compile("T[0-9]*_");
		ArrayList<ArrayList<String>> sentences = new ArrayList<>();
		ArrayList<String> gdepList = WFileReader.getFileLine(gdepFile);
		ArrayList<String> oneSenList = new ArrayList<>();
		for(String s : gdepList) {
			if(s.length() != 0) oneSenList.add(s);
			else {
				sentences.add(oneSenList);
				oneSenList = new ArrayList<>();
			}
		}
		
		ArrayList<String> a1List = WFileReader.getFileLine(a1File);
		ArrayList<String> outSampleList = new ArrayList<>();
		for(ArrayList<String> oneSen : sentences){
			Integer minOffset = Integer.valueOf(oneSen.get(0).split("\t")[8]);
			Integer maxOffset = Integer.valueOf(oneSen.get(oneSen.size()-1).split("\t")[9]);
			ArrayList<A1> a1InOneSen = new ArrayList<>();
			A1 a1Line = new A1();
			for(String s : a1List){
//				System.out.println(a1File.getAbsolutePath());
				a1Line.setName(s);
				if(a1Line.start >= minOffset && a1Line.end <= maxOffset) {
					//System.out.println(a1Line.biaohao);
					a1InOneSen.add(a1Line);
					a1Line = new A1();
				}
			}
			//System.out.println(a1InOneSen.size());
			HashSet<String> hasOffsetCross = new HashSet<>();
			for (int i = 0; i < a1InOneSen.size(); i++) {
				for (int j = i + 1; j < a1InOneSen.size(); j++){
					A1 entity1 = a1InOneSen.get(i);
					A1 entity2 = a1InOneSen.get(j);
					long start1 = entity1.start;
					long end1 = entity1.end;
					long start2 = entity2.start;
					long end2 = entity2.end;
					if (start1<=start2 & end2<=end1) {
						
						hasOffsetCross.add(entity1.biaohao + "_" + entity2.biaohao);
					}
					else if (start1>=start2 & end2>=end1) {
						
						hasOffsetCross.add(entity1.biaohao + "_" + entity2.biaohao);
					}
					else if (start1<start2 & end2>end1 & end1>start2) {
						
						hasOffsetCross.add(entity1.biaohao + "_" + entity2.biaohao);
					}
					else if (start1>start2 & end2<end1 & end2>start1) {
						hasOffsetCross.add(entity1.biaohao + "_" + entity2.biaohao);
					}
				}
			}
			//System.out.println(hasOffsetCross.size());
			if(a1InOneSen.size() >= 2) {
				for (int i = 0; i < a1InOneSen.size(); i++) {
					for (int j = i + 1; j < a1InOneSen.size(); j++) {
						A1 entity1 = a1InOneSen.get(i);
						A1 entity2 = a1InOneSen.get(j);
						//System.out.println(entity1.biaohao);
						String flag = entity1.biaohao + "_" + entity2.biaohao;
						//System.out.println(flag);
						//System.out.println(hasOffsetCross.size());
						if(!hasOffsetCross.contains(flag)) {
							//System.out.println("---");
							StringBuffer oneSample = new StringBuffer();
							for (int k = 0; k < oneSen.size(); k++) {
								String s = oneSen.get(k);
								int start = Integer.valueOf(s.split("\t")[8]);
								int end = Integer.valueOf(s.split("\t")[9]);
								//System.out.println(end);
								String word = "";
								if (start==entity1.start & end==entity1.end) {
									word = entity1.biaohao + "_" + entity1.type + "__" + entity1.word;
								}else if (start==entity2.start & end==entity2.end) {
									word = entity2.biaohao + "_" + entity2.type + "__" + entity2.word;
								}else if (start>=entity1.start & end<entity1.end) {						
									while(end < entity1.end){
										s = oneSen.get(++k);
										start = Integer.valueOf(s.split("\t")[8]);
										end = Integer.valueOf(s.split("\t")[9]);
									}
									word = entity1.biaohao + "_" + entity1.type + "__" + entity1.word;
									//System.out.println("%%" + entity1.word);
								}else if (start>=entity2.start & end<entity2.end) {
									while(end < entity2.end){
										s = oneSen.get(++k);
										start = Integer.valueOf(s.split("\t")[8]);
										end = Integer.valueOf(s.split("\t")[9]);
									}
									word = entity2.biaohao + "_" + entity2.type + "__" + entity2.word;
									
								}else {
									word = s.split("\t")[1];
								}
								oneSample.append(word + " ");
							}
							Matcher m = p.matcher(oneSample.toString());
							int count = 0;
							while (m.find()) {
								count++;
							}
							if(count==2) outSampleList.add(oneSample.toString());
						}
					}
				}
			}//if
		} //for
		WFileWriter.writeArrayListToFile(outSampleList, new File(gdepFile.getAbsolutePath().replace("\\gdep_process\\", "\\a2_1\\")));
		
		ArrayList<String> a2List = WFileReader.getFileLine(a2File);
		ArrayList<String> newA2 = new ArrayList<>();
		EventClass ec = new EventClass();
		
		for(String s : outSampleList) {
			int flag = 0; //判断是否为负例，若为0，则为负例
			for(String ss : a2List) {
				String sss[] = ss.split("\t")[1].split(" ");
				int type = ec.getClassIndex(sss[0]);
				String element1 = sss[1].split(":")[1];
				String element2 = sss[2].split(":")[1];
				if (s.contains(element1+"_") & s.contains(element2+"_")) {
					//newA2.add(type + " " + element1 + "_" + element2 + " "+ s);
					
					++flag;
					
					int index1 = s.indexOf(element1 + "_");
					int index2 = s.indexOf(element2 + "_");
					if(index1<index2) {
						int index = s.indexOf(" ", index2);
						if(index == -1 ) s = s.substring(index1);
						else s = s.substring(index1, index);
					}else {
						int index = s.indexOf(" ", index1);
						if(index == -1 ) s = s.substring(index2);
						else s = s.substring(index2, index);
					}
					newA2.add(type + " " + resetString2(s));
					allSample.add(type + " " + resetString(s) + " Distance_" + (s.split(" ").length-2));	
					//System.out.println(type + " " + "Distance_" + (s.split(" ").length-2));
					//allSample.add(0 + " " + resetString(s));	
				}
			}
			if(flag == 0) {
				Matcher m = p.matcher(s);
				String pair = "";
		        while (m.find()) {
		        	pair += m.group();
		        }
				//newA2.add(0 + " " + pair.substring(0, pair.length()-1) + " " + s);
				
		        
				int index1 = s.indexOf(pair.split("_")[0] + "_");
				int index2 = s.indexOf(pair.split("_")[1] + "_");
				if(index1<index2) {
					int index = s.indexOf(" ", index2);
					if(index == -1 ) s = s.substring(index1);
					else s = s.substring(index1, index);
				}else {
					int index = s.indexOf(" ", index1);
					if(index == -1 ) s = s.substring(index2);
					else s = s.substring(index2, index);
				}
				newA2.add(0 + " " + resetString2(s));
				allSample.add(0 + " " + resetString(s) + " Distance_" + (s.split(" ").length-2));
				//System.out.println("Distance_" + (s.split(" ").length-2));
			}
		} //for
		//System.out.println(newA2.size());
		WFileWriter.writeArrayListToFile(newA2, new File(gdepFile.getAbsolutePath().replace("\\gdep_process\\", "\\a2_2\\")));
	}
	
	public static String resetString(String s) {
		String ss[] = s.split(" ");
		String start = ss[0];
		String end = ss[ss.length-1];
		String type1 = "T_" + start.split("__")[0].split("_")[1];
		String type2 = "T_" + end.split("__")[0].split("_")[1];
		s = start.split("__")[1].replace("_", " ").replace("-", " ") + " ";
		String t = "";
		for (int i = 1; i < ss.length-1; i++) {
			t += ss[i] + " ";
		}
		s = s + t + end.split("__")[1].replace("_", " ").replace("-", " ") + " " + type1 + " " + type2;
		return s;
	}
	
	public static String resetString2(String s) {
		String ss[] = s.split(" ");
		String start = ss[0];
		String end = ss[ss.length-1];
		String type1 = start.split("__")[0];
		String type2 = end.split("__")[0];

		s = type1 + " " + type2;
		return s;
	}
	
	public static int setDistance(String s) {
		
		String ss[] = s.split(" ");		
		return 0;
	}
}
