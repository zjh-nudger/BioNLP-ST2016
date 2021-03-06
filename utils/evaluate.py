#coding=gbk
"""
Created on Sat Apr 18 09:36:37 2015

@author: zjh
"""
import numpy,csv

label_count={1:43,2:98,3:305,4:56,5:36,6:23,7:10,8:4,
             9:132,10:7,11:4,12:3,13:1,14:133,15:56,
             16:178,17:312,18:223,19:175} #实际的标注数量
span_sorted= sorted(label_count.iteritems(), key=lambda d:d[1], 
                                reverse = True)   
label_count_seedev = {1:55,2:23,3:9,4:20,5:29,6:179,7:0,8:10,
                      9:32,10:47,11:15,12:16,13:24,14:81,15:29,
                      16:13,17:59,18:111,19:0,20:39,21:8,22:20}
label_count_seedev_21 = {1:55,2:82,3:9,4:29,5:179,6:0,7:10,
                      8:32,9:47,10:15,11:16,12:24,13:81,14:29,
                      15:13,16:59,17:111,18:0,19:8,20:20}

#zjh{1: 43, 2: 89, 3: 291, 4: 56, 5: 36, 6: 20, 7: 8, 
#    8: 4, 9: 124, 10: 7, 11: 3, 12: 3, 13: 1, 14: 125,
#    15: 55, 16: 174, 17: 298, 18: 227, 19: 145}

#wy{1: 42, 2: 87, 3: 288, 4: 56, 5: 33, 6: 20,
#   7: 2, 8: 4, 9: 127, 10: 7, 11: 3, 12: 3, 
#   13: 1, 14: 113, 15: 52, 16: 174, 17: 290, 
#   18: 212, 19: 137}

def load_answer(answer):
    if len(numpy.shape(answer))==1:
        trigger_answer=answer.tolist()
        answer_list=[]
        for line in trigger_answer:
            answer_list.append(int(line))
        return answer_list
    else:
        trigger_answer=answer.tolist()
        answer_list=[]
        for line in trigger_answer:
            line=line[0]
            answer_list.append(int(line[0]))
        return answer_list

def load_prediction(prediction):
    if len(numpy.shape(prediction))==1:
        trigger_predict=prediction.tolist()
        predict_list=[]
        for line in trigger_predict:
            predict_list.append(int(line))
        return predict_list
    else:
        trigger_predict=prediction.tolist()
        predict_list=[]
        for line in trigger_predict:
            line=line[0]
            predict_list.append(int(line[0]))
        return predict_list
        
def calculate(tp,pre_pos,real_pos):
    precision=tp/(pre_pos+0.000001)
    recall=tp/(real_pos+0.0000001)
    f_score=2.0 * ((precision * recall)/(precision + recall+0.000001))
    return precision,recall,f_score   
    
def evaluate_single_class_seedev(prediction,answer,claz = 2):          
    list_answer=load_answer(answer)
    list_predict=load_prediction(prediction)
    assert len(list_answer) == len(list_predict)
    tp = 0
    pre_pos = 0
    real_pos = 0
    for answer,predict in zip(list_answer,list_predict):
        if answer == predict and predict == claz:
            tp += 1
        if predict == claz:
            pre_pos += 1
        if answer == claz:
            real_pos += 1
    precision,recall,f_score = calculate(tp,pre_pos,real_pos)
    return precision,recall,f_score

'''
def evaluate_multi_class_seedev(prediction=None,answer=None,claz_count = 2):
    list_answer=load_answer(answer)
    list_predict=load_prediction(prediction)
    f_score_micro_avg_1=[]
    f_score_micro_avg_2=[]
    f_score_macro_avg=[]
    precision_list = []
    recall_list = []
    all_tp = 0
    all_fp = 0
    for num in xrange(claz_count-1):
        num += 1
        tp = 0
        fp = 0
        for index in xrange(len(list_answer)-1):
            p = int(list_predict[index])
            d = int(list_answer[index])
            if p == d and p == num:
                tp += 1
                all_tp += 1
            if p > 0 and p != d:
                all_fp += 1
            if p == num and p != d:
                fp += 1
        precision,recall,f_score = calculate(tp,(fp+tp),label_count_seedev_21[num]) 
        precision_list.append(precision)
        recall_list.append(recall)
        f_score_macro_avg.append(f_score)
        f_score_micro_avg_1.append(precision * recall)
        f_score_micro_avg_2.append(precision + recall)
    #print len(f_score_mean)
    micro_f_score = 2*numpy.sum(f_score_micro_avg_1)/(numpy.sum(f_score_micro_avg_2)+0.00001)
    macro_f_score = numpy.mean(f_score_macro_avg)
#    if micro_f_score > 0.3:
#        for i in xrange(claz_count-1):
#            print precision_list[i],recall_list[i],f_score_macro_avg[i]
    all_precision = float(all_tp)/(all_fp+all_tp+0.00000001)
    event_count = 0
    for k,v in label_count_seedev_21.iteritems():
        event_count += v
    all_recall = float(all_tp) / event_count
    all_f1_score = 2.0 * ((all_precision * all_recall)/(all_precision + all_recall+0.000001))
    return all_precision,all_recall,all_f1_score#micro_f_score,macro_f_score
'''
def evaluate_multi_class_seedev(prediction=None,answer=None,claz_count = 2):
    list_answer=load_answer(answer)
    list_predict=load_prediction(prediction)
    all_tp = 0
    all_fp = 0
    for index in xrange(len(list_answer)-1):
        p = int(list_predict[index])
        d = int(list_answer[index])
        if p == d and p != 0:
            all_tp += 1
        if p > 0 and p != d:
            all_fp += 1
    all_precision = float(all_tp)/(all_fp+all_tp+0.00000001)
    event_count = 0
    for k,v in label_count_seedev.iteritems():
        event_count += v
    all_recall = float(all_tp) / event_count
    all_f1_score = 2.0 * ((all_precision * all_recall)/(all_precision + all_recall+0.000001))
    return all_precision,all_recall,all_f1_score#micro_f_score,macro_f_score


'''
将每个类别的准确率、召回率、F1值输出到文件
'''
def complete_evaluate(prediction,answer,csv_output):
    def calculate(tp,fp,label_num):
        precision=tp/(tp+fp+0.000001)
        recall=tp/(label_num+0.0000001)
        f_score=2.0 * ((precision * recall)/(precision + recall+0.000001))
        return precision,recall,f_score
    def load_answer(answer):
        if len(numpy.shape(answer))==1:
            trigger_answer=answer.tolist()
            answer_list=[]
            for line in trigger_answer:
                answer_list.append(int(line))
            return answer_list
        else:
            trigger_answer=answer.tolist()
            answer_list=[]
            for line in trigger_answer:
                line=line[0]
                answer_list.append(int(line[0]))
            return answer_list

    def load_prediction(prediction):
        if len(numpy.shape(prediction))==1:
            trigger_predict=prediction.tolist()
            predict_list=[]
            for line in trigger_predict:
                predict_list.append(int(line))
            return predict_list
        else:
            trigger_predict=prediction.tolist()
            predict_list=[]
            for line in trigger_predict:
                line=line[0]
                predict_list.append(int(line[0]))
            return predict_list
             
    list_answer=load_answer(answer)
        
    list_predict=load_prediction(prediction)
    
    csv_writer=csv.writer(open(csv_output,'wb'))
    csv_writer.writerow(['precision','recall','f_score'])
    
    f_score_micro_avg_1=[]
    f_score_micro_avg_2=[]
    f_score_macro_avg=[]
    for num in xrange(19):
        num+=1
        tp=0
        fp=0
        
        for index in xrange(len(list_answer)-1):
            p=int(list_predict[index])
            d=int(list_answer[index])
            if p==d and p==num:
                tp+=1
            if p==num and p!=d:
                fp+=1
        precision,recall,f_score=calculate(tp,fp,label_count[num])
        csv_writer.writerow([precision,recall,f_score])
        
        f_score_micro_avg_1.append(precision*recall)
        f_score_micro_avg_2.append(precision+recall)        
        if label_count[num]>=10:
            f_score_macro_avg.append(f_score)

    #print len(f_score_mean)
    micro_f_score=2*numpy.sum(f_score_micro_avg_1)/(numpy.sum(f_score_micro_avg_2)+0.00001)
    macro_f_score=numpy.mean(f_score_macro_avg)

    return micro_f_score,macro_f_score

'''
返回微平均以及宏平均
'''
def simple_evaluate(prediction,answer):
    def calculate(tp,fp,label_num):
        precision=tp/(tp+fp+0.000001)
        recall=tp/(label_num+0.0000001)
        f_score=2.0 * ((precision * recall)/(precision + recall+0.000001))
        return precision,recall,f_score
    def load_answer(answer):
        if len(numpy.shape(answer))==1:
            trigger_answer=answer.tolist()
            answer_list=[]
            for line in trigger_answer:
                answer_list.append(int(line))
            return answer_list
        else:
            trigger_answer=answer.tolist()
            answer_list=[]
            for line in trigger_answer:
                line=line[0]
                answer_list.append(int(line[0]))
            return answer_list

    def load_prediction(prediction):
        if len(numpy.shape(prediction))==1:
            trigger_predict=prediction.tolist()
            predict_list=[]
            for line in trigger_predict:
                predict_list.append(int(line))
            return predict_list
        else:
            trigger_predict=prediction.tolist()
            predict_list=[]
            for line in trigger_predict:
                line=line[0]
                predict_list.append(int(line[0]))
            return predict_list
             
    list_answer=load_answer(answer)
        
    list_predict=load_prediction(prediction)
    f_score_micro_avg_1=[]
    f_score_micro_avg_2=[]
    f_score_macro_avg=[]
    for num in xrange(19):
        num+=1
        tp=0
        fp=0
        for index in xrange(len(list_answer)-1):
            p=int(list_predict[index])
            d=int(list_answer[index])
            if p==d and p==num:
                tp+=1
            if p==num and p!=d:
                fp+=1
        precision,recall,f_score=calculate(tp,fp,label_count[num])     
        if label_count[num]>=10:
            f_score_macro_avg.append(f_score)
            f_score_micro_avg_1.append(precision*recall)
            f_score_micro_avg_2.append(precision+recall)   

    #print len(f_score_mean)
    micro_f_score=2*numpy.sum(f_score_micro_avg_1)/(numpy.sum(f_score_micro_avg_2)+0.00001)
    macro_f_score=numpy.mean(f_score_macro_avg)
    return micro_f_score,macro_f_score


'''
获得某个类别的f值
'''
def evaluate_class(prediction,answer,class_num):
    def calculate(tp,fp,label_num):
        precision=tp/(tp+fp+0.000001)
        recall=tp/(label_num+0.0000001)
        f_score=2.0 * ((precision * recall)/(precision + recall+0.000001))
        return precision,recall,f_score
    def load_answer(answer):
        if len(numpy.shape(answer))==1:
            trigger_answer=answer.tolist()
            answer_list=[]
            for line in trigger_answer:
                answer_list.append(int(line))
            return answer_list
        else:
            trigger_answer=answer.tolist()
            answer_list=[]
            for line in trigger_answer:
                line=line[0]
                answer_list.append(int(line[0]))
            return answer_list

    def load_prediction(prediction):
        if len(numpy.shape(prediction))==1:
            trigger_predict=prediction.tolist()
            predict_list=[]
            for line in trigger_predict:
                predict_list.append(int(line))
            return predict_list
        else:
            trigger_predict=prediction.tolist()
            predict_list=[]
            for line in trigger_predict:
                line=line[0]
                predict_list.append(int(line[0]))
            return predict_list
             
    list_answer=load_answer(answer)
        
    list_predict=load_prediction(prediction)

    f_score=0.    
    

    for num in xrange(19):
        num+=1
        tp=0
        fp=0
        
        for index in xrange(len(list_answer)-1):
            p=int(list_predict[index])
            d=int(list_answer[index])
            if p==d and p==num:
                tp+=1
            if p==num and p!=d:
                fp+=1
        precision,recall,f_score1=calculate(tp,fp,label_count[num])
        if class_num==num:
            f_score=f_score1
    return f_score


def weight_evaluate(prediction_p,answer):
    weight=numpy.ones(20)
    weight[19]=3
    weight[18]=1
    weight[17]=1
    prediction = prediction_p * weight
    prediction=numpy.argmax(prediction,axis=1)
    return simple_evaluate(prediction,answer)
    
    
def weight_complete_evaluate(prediction_p,answer,csv_output):
    weight=numpy.ones(20)
    weight[19]=10
    weight[18]=1
    weight[17]=1
    prediction = prediction_p * weight
    prediction=numpy.argmax(prediction,axis=1)
    return complete_evaluate(prediction,answer,csv_output)

def argument_simple_evaluate(prediction,answer):
    def calculate(tp,fp,label_num):
        precision=tp/(tp+fp+0.000001)
        recall=tp/(label_num+0.0000001)
        f_score=2.0 * ((precision * recall)/(precision + recall+0.000001))
        return precision,recall,f_score
    def load_answer(answer):
        if len(numpy.shape(answer))==1:
            trigger_answer=answer.tolist()
            answer_list=[]
            for line in trigger_answer:
                answer_list.append(int(line))
            return answer_list
        else:
            trigger_answer=answer.tolist()
            answer_list=[]
            for line in trigger_answer:
                line=line[0]
                answer_list.append(int(line[0]))
            return answer_list

    def load_prediction(prediction):
        if len(numpy.shape(prediction))==1:
            trigger_predict=prediction.tolist()
            predict_list=[]
            for line in trigger_predict:
                predict_list.append(int(line))
            return predict_list
        else:
            trigger_predict=prediction.tolist()
            predict_list=[]
            for line in trigger_predict:
                line=line[0]
                predict_list.append(int(line[0]))
            return predict_list
             
    list_answer=load_answer(answer)
        
    list_predict=load_prediction(prediction)

    tp=0
    fp=0
    for index in xrange(len(list_answer)-1):
        p=int(list_predict[index])
        d=int(list_answer[index])
        if p==d and p==1:
            tp+=1
        if p==d and p==2:
            tp+=1
        if p==2 and p!=d:
            fp+=1
        if p==1 and p!=d:
            fp+=1

    precision,recall,f_score=calculate(tp,fp,3724)      

    return precision,recall,f_score
