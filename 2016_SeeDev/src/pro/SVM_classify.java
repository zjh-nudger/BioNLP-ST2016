package pro;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class SVM_classify {
	
	public static void main(String[] args) throws IOException{
		String root="..\\svm_multiclass_windows\\";
		svm_classify(root +"svm_multiclass_classify", root + "data\\dev_svm", root + "data\\Mine_model", root + "data\\Mine_predict", 1);
	}
	
	public static void svm(){
		String root="..\\svm_multiclass_windows\\";
		svm_classify(root +"svm_multiclass_classify", root + "data\\dev_svm", root + "data\\Mine_model", root + "data\\Mine_predict", 1);
	}
	
	public static void svm_classify(String svmCommandname, String testfilename, String modelname, String prodicResultname, int flag) {  
        Runtime rt = Runtime.getRuntime(); 
        String commandStr = svmCommandname + " " + testfilename + " " + modelname + " " + prodicResultname;
        try {         
            Process pr = rt.exec(commandStr); //����cmd����  
            BufferedReader br = new BufferedReader(new InputStreamReader(pr.getInputStream()));  
            String s = br.readLine();  
            String temp = "" ;
            while(null != s ){  
            	if(!"".equals(s.trim()))  temp = s;                 
                s = br.readLine();  
            }  
            br.close();  
          
            if(flag == 1){
	            String sbustree = temp.substring(temp.indexOf(":") + 1).trim();
	            String[] data = sbustree.split("/");
	            double p = Double.parseDouble(data[0].substring(0,data[0].indexOf("%")).trim());
	            double r = Double.parseDouble(data[1].substring(0,data[1].indexOf("%")).trim());
	            double f = (p * r * 2) / (p + r);
	            temp = "<svm> " + temp + "  , F = " + f + "%";
	            System.out.println(temp);
            }else{
            	 System.out.println(temp);
            }
            //���µ�ǰ�̵߳ȴ�����Ҫ��һֱҪ�ȵ��ɸ� Process �����ʾ�Ľ���Ѿ���ֹ��  
            pr.waitFor();  
        } catch (IOException e) {  
        	//logger.error(e.getMessage());
            e.printStackTrace();  
        } catch (InterruptedException e) {
        	//logger.error(e.getMessage());
            e.printStackTrace();  
        }  
    }  
}
