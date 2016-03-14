package pro;

import java.io.File;
import java.util.ArrayList;

import util.EventClass;
import util.WFileReader;
import util.WFileWriter;

public class Postpro {

	public static void getA2Answer(File predicFile, File sampleFile) {
		
		File sampleFiless[] = sampleFile.listFiles();
		ArrayList<String> predic = WFileReader.getFileLine(predicFile);
		ArrayList<String> predicList = new ArrayList<>();
		String ss[] = predic.get(843).split(",");
		for (int i = 0; i < ss.length; i++) {
			predicList.add(ss[i]);
		}
		
		EventClass ec = new EventClass();
		int count = 0;
		int all = 0;
		for (int i = 0; i < sampleFiless.length; i++) {
			ArrayList<String> oneFileList = WFileReader.getFileLine(sampleFiless[i]);
			ArrayList<String> oneOutList = new ArrayList<String>();
			int sum = 0;
			//System.out.println(sampleFiless[i].getAbsolutePath());
			for (int j = 0; j < oneFileList.size(); j++) {
				
				if (!ec.getClassName(Integer.valueOf(predicList.get(count))).startsWith("Not")) {
					
					int index = oneFileList.get(j).indexOf(" ") + 1;
					String type = ec.getClassName(Integer.valueOf(predicList.get(count)));
					String t[] = oneFileList.get(j).substring(index).split(" ");
					String element1 = t[0].substring(t[0].indexOf("_")+1) + ":" + t[0].split("_")[0];
					String element2 = t[1].substring(t[1].indexOf("_")+1) + ":" + t[1].split("_")[0];
					if (amendmentAnswer(type, element1, element2).length() != 0) {
						all++;
						oneOutList.add("E" + ++sum + "\t" + type + " " + amendmentAnswer(type, element1, element2));
					}
//					amendmentAnswer(type, element1, element2);
//					oneOutList.add("E" + ++sum + "\t" + type + " " + element1 + " " + element2);
				}
				count++;
			}
			WFileWriter.writeArrayListToFile(oneOutList, new File(sampleFiless[i].getAbsolutePath().replace("\\a2_2\\", "\\a2_predict\\").replace("txt.gdep", "a2").replace("full", "binary")));
		}
		System.out.println(all);
	}
	
	public static String amendmentAnswer(String type, String element1, String element2) {
		
		switch (type) {
		case "Is_Member_Of_Family": is_Member_Of_Family(element1, element2); //1
			
		case "Is_Linked_To": return "Element1:" + element1.split(":")[1] + " " + "Element2:" + element2.split(":")[1]; //2
		
		case "Regulates_Tissue_Development": return regulates_Tissue_Development(element1, element2);	//3
		
		case "Has_Sequence_Identical_To": return "Element1:" + element1.split(":")[1] + " " + "Element2:" + element2.split(":")[1]; //4
		
		case "Is_Protein_Domain_Of": return is_Protein_Domain_Of(element1, element2); //5
			
		case "Regulates_Process": return regulates_Process(element1, element2); //6
		
		case "Composes_Protein_Complex": return composes_Protein_Complex(element1, element2); //7
		
		case "Occurs_During": return occurs_During(element1, element2); //8
		
		case "Interacts_With": return interacts_With(element1, element2); //9

		case "Is_Localized_In": return is_Localized_In(element1, element2); //10
		
		case "Composes_Primary_Structure": return composes_Primary_Structure(element1, element2); //11
		
		case "Occurs_In_Genotype": return occurs_In_Genotype(element1, element2); //12
		
		case "Binds_To": return binds_To(element1, element2); //13
		
		case "Exists_In_Genotype": return exists_In_Genotype(element1, element2); //14
		
		case "Regulates_Accumulation": return regulates_Accumulation(element1, element2); //15
		
		case "Transcribes_Or_Translates_To": return transcribes_Or_Translates_To(element1, element2); //16
		
		case "Regulates_Development_Phase": return regulates_Development_Phase(element1, element2); //17
		
		case "Regulates_Expression": return regulates_Expression(element1, element2); //18
		
		case "Regulates_Molecule_Activity": return regulates_Molecule_Activity(element1, element2); //19
		
		case "Is_Functionally_Equivalent_To": return "Element1:" + element1.split(":")[1] + " " + "Element2:" + element2.split(":")[1]; //20
		
		case "Exists_At_Stage": return exists_At_Stage(element1, element2); //21
		
		case "Is_Involved_In_Process": return is_Involved_In_Process(element1, element2); //21
		default: return "";
		}
	}


	private static String is_Involved_In_Process(String element1, String element2) {
		if((element1+" "+element2).contains("Regulatory_Network") | (element1+" "+element2).contains("Pathway")) {
			if(element1.contains("Regulatory_Network") | element1.contains("Pathway")) 
				switch (element2) {				
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Participant:" + element2.split(":")[1] + " " + "Process:" + element1.split(":")[1];
				default: return "";
				}
			else 
				switch (element1) {				
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Participant:" + element1.split(":")[1] + " " + "Process:" + element2.split(":")[1];
				default: return "";
				}
		} else return "";
	}

	private static String exists_At_Stage(String element1, String element2) {
		if((element1+" "+element2).contains("Development_Phase")) {
			if(element1.contains("Development_Phase")) {
				switch (element2) {					
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Functional_Molecule:" + element2.split(":")[1] + " " + "Development:" + element1.split(":")[1];
				default: return "";
				}
			} else {
				switch (element1) {				
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Functional_Molecule:" + element1.split(":")[1] + " " + "Development:" + element2.split(":")[1];
				default: return "";
				}
			}
		} else return "";
	}

	private static String regulates_Molecule_Activity(String element1, String element2) {
		switch (element1.split(":")[0]) {
		case "Gene":
		case "Gene_Family":
		case "Box":
		case "Promoter": 
		case "RNA":						
		case "Protein":
		case "Protein_Family":
		case "Protein_Complex":
		case "Protein_Domain":
		case "Hormone": return "Agent:"+element2.split(":")[1] + " " + "Molecule:" + element1.split(":")[1];
		case "Regulatory_Network":
		case "Pathway":
		case "Genotype":
		case "Tissue":
		case "Development_Phase":
		case "Environmental_Factor":
			switch (element2.split(":")[0]) {
			case "Gene":
			case "Gene_Family":
			case "Box":
			case "Promoter":
			case "RNA":						
			case "Protein":
			case "Protein_Family":
			case "Protein_Complex":
			case "Protein_Domain":
			case "Hormone": return "Agent:"+element1.split(":")[1] + " " + "Molecule:" + element2.split(":")[1];
			default: return "";
			}
		default: return "";
		}	
	}

	private static String regulates_Expression(String element1, String element2) {
		switch (element1.split(":")[0]) {
		case "Gene":
		case "Gene_Family":
		case "Box":
		case "Promoter": return "Agent:"+element2.split(":")[1] + " " + "DNA:" + element1.split(":")[1];
		case "RNA":						
		case "Protein":
		case "Protein_Family":
		case "Protein_Complex":
		case "Protein_Domain":
		case "Hormone":
		case "Regulatory_Network":
		case "Pathway":
		case "Genotype":
		case "Tissue":
		case "Development_Phase":
		case "Environmental_Factor":
			switch (element2.split(":")[0]) {
			case "Gene":
			case "Gene_Family":
			case "Box":
			case "Promoter": return "Agent:"+element1.split(":")[1] + " " + "DNA:" + element2.split(":")[1];
			default: return "";
			}
		default: return "";
		}		
	}

	private static String regulates_Development_Phase(String element1, String element2) {
		if((element1+" "+element2).contains("Development_Phase")) {
			if(element1.contains("Development_Phase")) return "Agent:" + element2.split(":")[1] + " " + "Development:" + element1.split(":")[1];
			else return "Agent:" + element1.split(":")[1] + " " + "Development:" + element2.split(":")[1];
		} else return "";
	}

	private static String transcribes_Or_Translates_To(String element1, String element2) {
		switch (element1.split(":")[0]) {
		case "RNA":						
		case "Protein":
		case "Protein_Family":
		case "Protein_Complex":
		case "Protein_Domain": 
			switch (element2.split(":")[0]) {
			case "RNA":						
			case "Gene":
			case "Gene_Family":
			case "Box":
			case "Promoter": return "Source:"+element1.split(":")[1] + " " + "Product:" + element2.split(":")[1];
			default: return "";
			}					
		case "Gene":
		case "Gene_Family":
		case "Box":
		case "Promoter": 
			switch (element2.split(":")[0]) {
			case "RNA":						
			case "Protein":
			case "Protein_Family":
			case "Protein_Complex":
			case "Protein_Domain": return "Source:"+element2.split(":")[1] + " " + "Product:" + element1.split(":")[1];
			default: return "";
			}
		default: return "";
		}
	}

	private static String regulates_Accumulation(String element1, String element2) {
		switch (element1.split(":")[0]) {
		case "RNA":						
		case "Protein":
		case "Protein_Family":
		case "Protein_Complex":
		case "Protein_Domain":
		case "Hormone": return "Agent:"+element2.split(":")[1] + " " + "Function_Molecule:" + element1.split(":")[1];
		case "Gene":
		case "Gene_Family":
		case "Box":
		case "Promoter": 
		case "Regulatory_Network":
		case "Pathway":
		case "Genotype":
		case "Tissue":
		case "Development_Phase":
		case "Environmental_Factor":
			switch (element2.split(":")[0]) {
			case "RNA":						
			case "Protein":
			case "Protein_Family":
			case "Protein_Complex":
			case "Protein_Domain":
			case "Hormone": return "Agent:"+element1.split(":")[1] + " " + "Function_Molecule:" + element2.split(":")[1];
			default: return "";
			}
		default: return "";
		}		
	}

	private static String exists_In_Genotype(String element1, String element2) {
		if((element1+" "+element2).contains("Genotype")) {
			if (element1.contains("Genotype")) {
				switch (element2) {
				case "Genotype":
				case "Tissue":
				case "Development_Phase": return "Element:" + element2.split(":")[1] + " " + "Genotype:" + element1.split(":")[1];
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Molecule:" + element2.split(":")[1] + " " + "Genotype:" + element1.split(":")[1];
				default: return "";
				}
			} else {
				switch (element1) {
				case "Genotype":
				case "Tissue":
				case "Development_Phase": return "Element:" + element1.split(":")[1] + " " + "Genotype:" + element2.split(":")[1];
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Molecule:" + element1.split(":")[1] + " " + "Genotype:" + element2.split(":")[1];
				default: return "";
				}
			}
		} else return "";
	}

	private static String binds_To(String element1, String element2) {
		switch (element1.split(":")[0]) {
		case "RNA":						
		case "Protein":
		case "Protein_Family":
		case "Protein_Complex":
		case "Protein_Domain":
		case "Hormone":
			switch (element2.split(":")[0]) {
			case "Gene":
			case "Gene_Family":
			case "Box":
			case "Promoter":
			case "RNA":						
			case "Protein":
			case "Protein_Family":
			case "Protein_Complex":
			case "Protein_Domain":
			case "Hormone": return "Functional_Molecule:"+element1.split(":")[1] + " " + "Molecule:" + element2.split(":")[1];
			default: return "";
			}
		case "Gene":
		case "Gene_Family":
		case "Box":
		case "Promoter": 
			switch (element2.split(":")[0]) {
			case "RNA":						
			case "Protein":
			case "Protein_Family":
			case "Protein_Complex":
			case "Protein_Domain":
			case "Hormone": return "Functional_Molecule:"+element2.split(":")[1] + " " + "Molecule:" + element1.split(":")[1];
			default: return "";
			}
		default: return "";
		}
	}

	private static String occurs_In_Genotype(String element1, String element2) {
		if((element1+" "+element2).contains("Genotype")) {
			if(element1.contains("Genotye")) {
				switch (element2) {					
				case "Regulatory_Network":
				case "Pathway": return "Genotye:" + element1.split(":")[1] + " " + "Process:" + element2.split(":")[1];
				default: return "";
				}
			} else {
				switch (element1) {				
				case "Regulatory_Network":
				case "Pathway": return "Genotye:" + element2.split(":")[1] + " " + "Process:" + element1.split(":")[1];
				default: return "";
				}
			}
		} else return "";
	}

	private static String composes_Primary_Structure(String element1, String element2) {
		if((element1+" "+element2).contains("Box") | (element1+" "+element2).contains("Promoter")) {
			if(element1.contains("Box") | element1.contains("Promoter")) {
				switch (element2) {					
				case "Gene":
				case "Gene_Family":
				case "Box":
				case "Promoter": return "DNA_Part:" + element1.split(":")[1] + " " + "DNA:" + element2.split(":")[1];
				default: return "";
				}
			} else {
				switch (element1) {				
				case "Gene":
				case "Gene_Family":
				case "Box":
				case "Promoter": return "DNA_Part:" + element2.split(":")[1] + " " + "DNA:" + element1.split(":")[1];
				default: return "";
				}
			}
		} else return "";
	}

	private static String is_Localized_In(String element1, String element2) {
		if((element1+" "+element2).contains("Tissue")) {
			if (element1.contains("Tissue")) {
				switch (element2) {
				case "Regulatory_Network":
				case "Pathway": return "Process:" + element2.split(":")[1] + " " + "Target_Tissue:" + element1.split(":")[1];
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Functional_Molecule:" + element2.split(":")[1] + " " + "Target_Tissue:" + element1.split(":")[1];
				default: return "";
				}
			} else {
				switch (element1) {
				case "Regulatory_Network":
				case "Pathway": return "Process:" + element1.split(":")[1] + " " + "Target_Tissue:" + element2.split(":")[1];
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Functional_Molecule:" + element1.split(":")[1] + " " + "Target_Tissue:" + element2.split(":")[1];
				default: return "";
				}
			}
		} else return "";
	}

	private static String interacts_With(String element1, String element2) {
		switch (element1.split(":")[0]) {
		case "Hormone": 
		case "Regulatory_Network":
		case "Pathway":
		case "Genotype":
		case "Tissue":
		case "Development_Phase":
		case "Environmental_Factor":
			switch (element2.split(":")[0]) {
			case "Gene":
			case "Gene_Family":
			case "Box":
			case "Promoter":
			case "RNA":						
			case "Protein":
			case "Protein_Family":
			case "Protein_Complex":
			case "Protein_Domain":
			case "Hormone": return "Agent:"+element1.split(":")[1] + " " + "Target:" + element2.split(":")[1];
			default: return "";
			}		
		default: return "Agent:"+element2.split(":")[1] + " " + "Target:" + element1.split(":")[1];
		}
	}

	private static String occurs_During(String element1, String element2) {
		if((element1+" "+element2).contains("Development_Phase")) {
			if(element1.contains("Development_Phase")) {
				switch (element2) {					
				case "Regulatory_Network":
				case "Pathway": return "Domain:" + element1.split(":")[1] + " " + "Product:" + element2.split(":")[1];
				default: return "";
				}
			} else {
				switch (element1) {				
				case "Regulatory_Network":
				case "Pathway": return "Domain:" + element2.split(":")[1] + " " + "Product:" + element1.split(":")[1];
				default: return "";
				}
			}
		} else return "";
	}

	private static String composes_Protein_Complex(String element1, String element2) {
		if((element1+" "+element2).contains("_Complex")) {
			if(element1.contains("_Complex")) {
				switch (element2) {					
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain": return "Domain:" + element1.split(":")[1] + " " + "Product:" + element2.split(":")[1];
				default: return "";
				}
			} else {
				switch (element1) {				
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain": return "Domain:" + element2.split(":")[1] + " " + "Product:" + element1.split(":")[1];
				default: return "";
				}
			}
		} else return "";
	}

	private static String is_Member_Of_Family(String element1, String element2) {
		if(element1.contains("_Family") & (element2.contains("Gene")|element2.contains("Protein"))) {
			return "Element:" + element2.split(":")[1] + " " + "Family:" + element1.split(":")[1];
		}
		else if(element2.contains("_Family") & (element1.contains("Gene")|element1.contains("Protein"))) {
			return "Element:" + element1.split(":")[1] + " " + "Family:" + element2.split(":")[1];
		}
		else if(element1.equals("RNA") & element2.equals("RNA")) {
			return "Element:" + element1.split(":")[1] + " " + "Family:" + element2.split(":")[1];
		}
		else return "";
	}
	
	private static String regulates_Tissue_Development(String element1, String element2) {
		if((element1+" "+element2).contains("Tissue")) {
			if(element1.contains("Tissue")) return "Agent:" + element2.split(":")[1] + " " + "Target_Tissue:" + element1.split(":")[1];
			else return "Agent:" + element1.split(":")[1] + " " + "Target_Tissue:" + element2.split(":")[1];
		} else return "";
	}
	
	private static String is_Protein_Domain_Of(String element1, String element2) {
		if((element1+" "+element2).contains("_Domain")) {
			if(element1.contains("_Domain")) {
				switch (element2) {
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Domain:" + element1.split(":")[1] + " " + "Product:" + element2.split(":")[1];
				default: return "";
				}
			} else {
				switch (element1) {
				case "RNA":						
				case "Protein":
				case "Protein_Family":
				case "Protein_Complex":
				case "Protein_Domain":
				case "Hormone": return "Domain:" + element2.split(":")[1] + " " + "Product:" + element1.split(":")[1];
				default: return "";
				}
			}
		} else return "";
	}
	
	private static String regulates_Process(String element1, String element2) {
		if((element1+" "+element2).contains("Regulatory_Network") | (element1+" "+element2).contains("Pathway")) {
			if(element1.contains("Regulatory_Network") | element1.contains("Pathway")) 
				return "Agent:" + element2.split(":")[1] + " " + "Process:" + element1.split(":")[1];
			else return "Agent:" + element1.split(":")[1] + " " + "Process:" + element2.split(":")[1];
		} else return "";
	}
}
