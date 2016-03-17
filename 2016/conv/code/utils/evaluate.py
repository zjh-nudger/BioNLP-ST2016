#coding=gbk
"""
"""
import numpy,csv

label_count={1:55,2:23,3:9,4:20,5:29,6:179,7:0,8:10,
             9:32,10:47,11:15,12:16,13:24,14:81,15:29,
             16:13,17:59,18:111,19:0,20:39,21:8,22:20}


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
    for num in xrange(22):
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


def simple_evaluate(prediction,answer,out):
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
    out.append(list_predict)
    f_score_micro_avg_1=[]
    f_score_micro_avg_2=[]
    f_score_macro_avg_1=[]
    f_score_macro_avg_2=[]
    f_score_macro_avg=[]
    each_R=[]
    each_P=[]
    each_F=[]
    all_tp=0
    all_fp=0
    all_label_count=0
    for num in xrange(22):
        num+=1
        tp=0
        fp=0
        
        for index in xrange(len(list_answer)-1):
            p=int(list_predict[index])
            d=int(list_answer[index])
            if p==d and p==num:
                tp+=1
                all_tp+=1
            if p==num and p!=d:
                fp+=1
                all_fp+=1
                
        all_label_count+=label_count[num]
        #每个类别的precision，recall，f值
        precision,recall,f_score=calculate(tp,fp,label_count[num])
        each_R.append(recall);
        each_P.append(precision);
        each_F.append(f_score);
        #zjh计算方法
        f_score_micro_avg_1.append(precision*recall)
        f_score_micro_avg_2.append(precision+recall)  
        f_score_macro_avg_1.append(precision)
        f_score_macro_avg_2.append(recall) 
        f_score_macro_avg.append(f_score)

    #我的计算方法
    #微平均
    a,b,micro_f_score=calculate(all_tp,all_fp,all_label_count)
    if(micro_f_score>=0.33):
        print all_tp,all_fp,all_label_count
        for i in xrange(22):        
            print i+1,each_R[i],each_P[i],each_F[i]
            i+=1
    #宏平均
    h1=numpy.mean(f_score_macro_avg_1)
    h2=numpy.mean(f_score_macro_avg_2)
    macro_f_score=2*h1*h2/(h1+h2+0.000001)

    #zjh
    micro=2*numpy.sum(f_score_micro_avg_1)/(numpy.sum(f_score_micro_avg_2)+0.00001)
    macro=numpy.mean(f_score_macro_avg)
    
    print a,b,micro_f_score,'-',h1,h2,macro_f_score,'-',micro,macro
    return micro_f_score,macro_f_score

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
        if p==1 and p!=d:
            fp+=1

    precision,recall,f_score=calculate(tp,fp,3724)      

    return precision,recall,f_score

if __name__=='__main__':
    print '\n'
    print complete_evaluate()