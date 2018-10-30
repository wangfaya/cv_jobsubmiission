# encoding=utf-8

import jieba #分词用的
import codecs#这个库用来解决编码问题
from collections import defaultdict#这个是字典库
import pandas as pd
import pymysql
import matplotlib
from tkinter import *
from tkinter import messagebox


#分词
def seg_word(contens):
    word=jieba.cut(contens)
    seg_result=[]
    for w in word:
        seg_result.append(w)
    stoplist=set()
    stopfile=codecs.open("./data/stoplist.txt","r","utf-8")
    for i in stopfile:
        stoplist.add(i.strip())
    stopfile.close()

    return list(filter(lambda x:x not in stoplist,seg_result))
#词语分类
def classfire_word(word_dict):
    """词语分类  找出情感词  否定词    程度副词"""
    #读取情感词典文件
    sen_file=open("./data/情感极性词典/BosonNLP_sentiment_score/BosonNLP_sentiment_score.txt","r+",encoding="utf-8")
    sen_list=sen_file.readlines()
    sen_dict=defaultdict()
    for w in sen_list:
        sen_dict[w.strip().split()[0]]=w.strip().split()[1]
    #读取否定词文件
    not_word_file=open("./data/情感极性词典/否定词.txt","r+",encoding="utf-8")
    not_word_file_list=not_word_file.readlines()
    not_word_list=[]
    for i in not_word_file_list:
        not_word_list.append(i.strip())

    #读取程度副词文件
    degree_file=open("./data/情感极性词典/程度副词.txt","r+",encoding="utf-8")
    degree_list=degree_file.readlines()
    degree_dict=defaultdict()
    for s in degree_list:
        degree_dict[s.strip().split(" ")[0]]=s.strip().split(" ")[1]

    sen_word=dict()
    not_word=dict()
    degree_word=dict()
    for word in word_dict.keys():
        # print(word)
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dict.keys():
            sen_word[word_dict[word]]=sen_dict[word]
        elif word in not_word_list and word not in degree_dict.keys():
            not_word[word_dict[word]]=-1
        elif word in degree_dict.keys():
            degree_word[word_dict[word]]=degree_dict[word]
    sen_file.close()
    not_word_file.close()
    degree_file.close()
    return sen_word,not_word,degree_word
#把分词后的单词转化为字典   健为单词本省  值为单词在列表中的索引
def list_to_dict(word_list):
    data={}
    for i in range(0,len(word_list)):
        data[word_list[i]]=i
    return data
#初始化权重  默认为1   前面有否定词乘以-1   有程度副词乘以程度副词的程度值
def init_weight(sen_word,not_word,degree_word):
    #初始化权重为1
    w=1
    sen_word_index_list=list(sen_word.keys())
    if len(sen_word_index_list)==0:
        return w
    for i in range(0,sen_word_index_list[0]):
        if i in not_word.keys():
            w*=-1
        elif i in degree_word.keys():
            w*=float(degree_word[i])
    return w
#计算分值
def score_sentiment(sen_word,not_word,degree_word,seg_result):
    w=init_weight(sen_word,not_word,degree_word)
    score=0
    #情感词下标设为-1
    sentiment_index=-1
    #情感词的位置下标集合
    sentiment_index_list=list(sen_word.keys())
    #遍历分词结果，目的为了找出分词结果中的否定词以及程度副词
    for i in range(0,len(seg_result)):
        #如果是情感词的话
        if i in sen_word.keys():
            score+=w*float(sen_word[i])
            sentiment_index+=1
            if sentiment_index<len(sentiment_index_list)-1:
                # 判断当前的情感词和下一个情感词之间有没有程度副词
                for j in range(sentiment_index_list[sentiment_index],sentiment_index_list[sentiment_index+1]):
                    if j in not_word.keys():
                        w*=-1
                    elif j in degree_word.keys():
                        w*=float(degree_word[j])
        #去下一个情感词
        if sentiment_index<len(sentiment_index_list)-1:
            i=sentiment_index_list[sentiment_index+1]
    return score
#开始计算得分
def sentiment_score(sentiment):
    word_list=seg_word(sentiment)
    sen_word,not_word,degree_word=classfire_word(list_to_dict(word_list))
    score=score_sentiment(sen_word,not_word,degree_word,word_list)
    # return score
    if -1 < score < 1:
        # res.set("中性")
        print('中性')
        return "中性"
    if score > 1:
        # res.set("积极")
        print('积极')
        return "积极"

    if score < -1:
        # res.set("消极")
        print('消极')
        return "消极"

    # return res
# def analyle():
#     # contens=entry.get()
#     if contens=="":
#         messagebox.showinfo("提示","请输入要分析的语句:")
#     else:
#         sentiment_score(contens)
# -----------------------------------------gui用户界面---------------------------------------------------------------------
# root=Tk()
# root.title("简易情感分析")
# root.geometry("360x100+200+300")
# label1=Label(root,text="请输入要分析的语句:")
# label1.grid(row=0,column=0)
# label2=Label(root,text="语句的态度为:")
# label2.grid(row=1,column=0)
# res=StringVar()
# entry=Entry(root,font=("微软雅黑",15))
# entry.grid(row=0,column=1)
# entry1=Entry(root,font=("微软雅黑",15),textvariable=res)
# entry1.grid(row=1,column=1)
# button=Button(root,text="分析",width=10,command=analyle)
# button.grid(row=2,column=0,sticky=W)
# button1=Button(root,text="退出",width=10,command=root.quit)
# button1.grid(row=2,column=1,sticky=E)
# button.pack
# button1.pack
# root.mainloop()
#----------------------------------------对爬取得数据内容进行分析------------------------------------------------------------
conn=pymysql.connect(host='localhost',user='root',passwd='root',db='腾讯招聘',port=3306,charset='utf8')
# data=pd.read_csv("./data/mamabang.csv",encoding='utf-8',error_bad_lines=False)
data=pd.read_sql("select * from mamabang where content like '%英语%'",conn)
print(data.head())
data=pd.DataFrame(data[['name','email','content']].astype(str))
# print(data['name'].head())
data_len=len(data['content'])
num=1
wordlist=[]
for content in (data['content']):
    print("当前正在分析第%d条，剩余%d条"%(num,data_len-num),end=' ')
    num+=1
    manner=sentiment_score(content)
    # print('态度为:%s'%manner)
    wordlist.append(manner)
data['manner']=wordlist




data.to_csv('dataanalyze_mamabang_English.csv')


#测试
# print(sentiment_score("本人励步英语幼儿组老师，毕业四年，做过英语老师汉语老师电台主播各种杂七杂八的工作，励步入职四个月，17年4月份找工作的时候最后纠结于两个offer，一个励步，另外一个某大培训机构的自考汉语教师的offer。最后选择了励步。没有在其他早教机构任职过，就少儿教师岗位发表一下自己的经验吧。先说最关心的薪资。总体来说在业界属于中高水平。幼儿组教师带6个班（最多带7个）平均月薪10K-15K。新入职老师前六个月有底薪，六个月之后完全靠课时费。励步的课时费计算非常复杂，根据学校学位专业工作年限证书等等计算出每节课的课时费，一般刚入职都是200多，教高阶的教师课时费更高。做好家长服务升班奖金也非常可观。想涨课时费主要可以多带班，去考各种证，做好绩效涨教师等级，提升空间非常大。再说对教师的要求。整个行业对于教师的要求都是越来越严的。励步也是。初试复试筛一批人，然后进入初级培训班，会淘汰一批，然后进入带培班，依然有可能淘汰。对少儿教师的要求当然：1，英语能力。比起托福教师雅思老师可能会低一些，但是不代表学过英语就可以教小朋友哦！2，教学能力。我认为这是最重要的一点。怎么把知识传授给学生，这是一种能力，和天赋和训练都有关，外向，热情的性格更适合做老师，教小朋友还需要你有一点演技，课堂的生动活泼的气氛非常重要！没有经验没关系，励步有将近6个月的带薪培训。这一点非常难得，对于老师的职业素养提升非常有帮助。3，亲和力和童心。这两点会让孩子喜欢上你。这也非常重要。当孩子觉得你无聊或者你的课堂无聊甚至不喜欢你害怕你的时候，成绩不好不说，家长也不会满意，不续费甚至退费非常打击信心，也不利于职业成长。4，服务意识。教育机构不同于公立学校，我们属于服务行业，意识上我们要做好老师和服务者的平衡。服务者就是做小伏低，骂不还手打不还口，如果心气高受不得气的小朋友要谨慎考虑服务行业。这还只是底线。把服务做好的意思在励步还包括课后的服务工作。发作业、评作业、家长联络沟通等等这些是工作内容。提学习意见、提升家长的教育理念等等这些是更高层的要求，需要慢慢去训练和提升。再说说带薪培训和带班。拿到offer后会安排三周的初级培训，主要内容包括企业文化，励步的课程体系，试听课，常规课等等的培训，每周都有考核，最后有一个分数，不达标的老师会被淘汰。通过的老师就可以进入带培阶段，这个阶段依然会有各种课程的培训，会有培训师手把手教你怎么上课，怎么服务家长，怎么和孩子相处，怎么管理你的课堂等等。这个阶段老师就可能带上班了。带上班后依然会有培训师手把手帮老师解决遇到的各种问题，这对于刚毕业的新手来说是一个非常难得的学习和进步过程。最后说说晋升空间。如果喜欢教学可以一直做教学，可以一直做到五星教师，去考一些英语能力或者教学能力方面的证，做好升班率，课时费也会很高很高。或者对管理感兴趣也可以往培训师方向走。少儿英语培训行业正是黄金年代，励步被好未来收购之后发展速度也是大步前进。所以对于教师的需求也是越来越多，励步随时都在招聘英语教师，随时可以投简历，可见需求之大。所以对于培训师的需求也会越来越多。或者对课程研发有兴趣也可以往相关方向发展。当初拿到两个offer。自考汉语教师的薪水更高，入职带上了班月薪就10K+，励步要入职大半年才能达到同样的薪资水平。但是最终选择了励步主要就是因为在励步有得成长有得发展"))
# sentiment_score("发信人:Yangdaisy(Yangdaisy),信区:EnglishWorld标题:[原创]上海有英语角吗发信站:水木社区(FriSep2117:19:542018),站内上海有英语角吗--※来源:·水木社区http://www.newsmth.net·[FROM:116.226.83.*]")
# a=sentiment_score("我不会不喜欢你")
# print(a)
# sentiment_score("发信人:weina042(weina042),信区:EnglishWorld标题:低价转让华尔街英语课程发信站:水木社区(WedFeb2816:51:362018),站内本人在华尔街英语学校学习一年多后，又续报了7个级别的课程，因为个人家庭原因不能再继续学习，现欲转让未学续报课程，因为华尔街英语每年会涨价几次，我这个课程是去年续保的，所以价格比现在同级数的价格优惠很多，在学习一年后，我的英语也长进不少，对我工作帮助很大，如有对英语学习感兴趣的亲可与我联系：13811990416（同微信）--※来源:·水木社区http://www.newsmth.net·[FROM:106.38.2.*]发信人:Harden13(眼神防守大师),信区:EnglishWorld标题:Re:低价转让华尔街英语课程发信站:水木社区(TueSep2510:59:452018),站内请问课程还在吗？【在weina042的大作中提到:】:本人在华尔街英语学校学习一年多后，又续报了7个级别的课程，因为个人家庭原因不能再继续学习，现欲转让未学续报课程，因为华尔街英语每年会涨价几次，我这个课程是去年续保的，所以价格比现在同级数的价格优惠很多，在学习一年后，我的英语也长进不少，对我工作帮助很大，如有对英语学习感兴趣的亲可与我联系：13811990416（同微信）--※来源:·水木社区http://www.newsmth.net·[FROM:123.127.137.*]")
# print(sentiment_score("我特别讨厌你"))sentiment_score("英语")

# for i in range(4):
#     contens=input("请输入要分析的语句：")
#     score=sentiment_score(contens)
#     if -1 < score <1  :
#         print("中性")
#     if score > 1:
#         print("积极")
#     if score < -1:
#         print("消极")
#     print("情感值为%s"%score)
#     print("*"*80)
# print(seg_word("大家都别去励步英语"))
# print(classfire_word(list_to_dict(seg_word("大家都别去励步英语，太差劲了"))))
