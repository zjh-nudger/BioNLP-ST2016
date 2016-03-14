package pro;

public class A1 {

	String biaohao = "";
	String type = "";
	long start = 0;
	long end = 0;
	String word = "";
	
	public void setName(String s) {
		if (s.contains(";")) {
			String ss[] = s.split("\t");
			this.biaohao = ss[0];
			this.type = ss[1].split(" ")[0];
			this.start = Integer.valueOf(ss[1].split(" ")[1]);
			this.end = Integer.valueOf(ss[1].split(" ")[ss[1].split(" ").length-1]);
			if (ss[2].contains(" ")) {
				String sss[] = ss[2].split(" ");
				StringBuffer sb = new StringBuffer();
				for (int i = 0; i < sss.length; i++) {
					sb.append(sss[i]);
					if(i < sss.length - 1) sb.append("_");
				}
				this.word = sb.toString();
			}else {
				this.word = ss[2];
			}
		}else {
			String ss[] = s.split("\t");
			this.biaohao = ss[0];
			this.type = ss[1].split(" ")[0];
			this.start = Integer.valueOf(ss[1].split(" ")[1]);
			this.end = Integer.valueOf(ss[1].split(" ")[2]);
			
			if (ss[2].contains(" ")) {
				String sss[] = ss[2].split(" ");
				StringBuffer sb = new StringBuffer();
				for (int i = 0; i < sss.length; i++) {
					sb.append(sss[i]);
					if(i < sss.length - 1) sb.append("_");
				}
				this.word = sb.toString();
			}else {
				this.word = ss[2];
			}
		}		
	}
}
