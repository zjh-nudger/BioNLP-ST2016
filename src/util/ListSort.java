/*
 * File: ListSort。java
 */

package util;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Collections;


public class ListSort{
	private static ComparatorUser comparator;
	@SuppressWarnings("unchecked")
	public static ArrayList<Integer> listSort(ArrayList<Integer> list){ 
		comparator = new ComparatorUser();
		Collections.sort(list, comparator);
		return list;
	}
}

@SuppressWarnings("rawtypes")
class ComparatorUser implements Comparator{
	/*
	 * 对list中的指定值进行排序返回
	 */
	public int compare(Object o1, Object o2){
		Integer temp1 = (Integer) o1;
		Integer temp2 = (Integer) o2;
		return temp1.compareTo(temp2);
	}
}
