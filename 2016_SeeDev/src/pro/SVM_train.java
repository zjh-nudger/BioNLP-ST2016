package pro;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class SVM_train {
	public static void main(String[] args) throws IOException{
		String root="..\\svm_multiclass_windows\\";
		svm_learn(root +"svm_multiclass_learn", root + "data\\train_svm", root + "data\\Mine_model");
	}
	
	public static void svm(){
		String root="..\\svm_multiclass_windows\\";
		svm_learn(root +"svm_multiclass_learn", root + "data\\train_svm", root + "Mine_model");
	}
	
	public static void svm_learn(String svmCommandname, String sourcefilename, String modelResultname) {  
        Runtime rt = Runtime.getRuntime();  
        String commandStr = svmCommandname + " " + sourcefilename + " " + modelResultname;
        //System.out.println(commandStr);
        try {         
            Process pr = rt.exec(commandStr); 
            BufferedReader br = new BufferedReader(new InputStreamReader(pr.getInputStream()));  
            String s = br.readLine(); 
            String temp = "";
            while(null != s ){  
                if(!"".equals(s.trim()))  
                	temp = temp + s;
                System.out.println(s);
                s = br.readLine();  
            }  
            br.close();  

            pr.waitFor(); 
        } catch (IOException e) {  
            e.printStackTrace();  
        } catch (InterruptedException e) {  
            e.printStackTrace();  
        }  
    } //svm_learn()
	
}//SVM_train
