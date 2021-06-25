from nltk.stem import WordNetLemmatizer
import os
import numpy as np
import pandas as pd
import math
from operator import itemgetter
import tkinter as tk
from tkinter import ttk

totaldocs=50
Alpha=0.005
#---------------------Dictionary of Posting List, Storing Similarity and to map a doc id------------------------
Posting_List={}
docIdMap={}
dict={}
Result={}

 #----------------------------------------- Pre Processing function for text---------------------------
def Pre_Processing(input_string):
    result=''
    #------------Removing Punctuations In Input Querry-----------
    punctuations=['.', ',','’','‘','“','ã','(', ')', "'", "!",'©', ':','?','"',';','#','&','*','@','[',']','-','/','{','}','$','“',',','”','1','2','3','4','5','6','7','8','9','0']
    for character in input_string:
        if character == '.':
           result=result + " "
           continue
        if character in punctuations:
            continue
        else:
            result=result + character
    
    GottenStopWords=getStopWords()
    result=removeStopWords(result,GottenStopWords)
    result=make_word_list(result)
    return result

 #----------------------------------------Removing Stopwords from Text------------------------------
def removeStopWords(input_string, StopWordss):
    line=input_string.split()
    result=""
    if line[0].isdigit():
        result.join(line)
    for word in line:
        if word not in StopWordss:
            result=result + (word + " ")
    return result

 #-------------------------------------Getting Stopwords from Given File------------------------------
def getStopWords():
    StopWords = open('Stopword-List.txt')
    stopWordssList = []
    for sw in StopWords:
        sw = sw.strip()
        stopWordssList.append(sw)
    return stopWordssList

 #----------------------------------------Tokeninzing the Doccument data------------------------------
def make_word_list(word_list):   
    W=[]
    w=''
    for word in word_list: 
       if((word!=' ')and(word!='.')and(word!=']')and(word!='\n')and(word!='-')and(word!='—')and(word!='?')and(word!='"')and(word!='…')and(word!='/')):
          w=w+word 
       elif((w!='') ):
            W=W+[w]  
            w=''
    if((w!='') ):
        W=W+[w]
    l=len(W)
    #---------Here lemmatization id done in tokens using nltk library--------
    lemmatizer = WordNetLemmatizer()
    for i in range(l):
        W[i]=lemmatizer.lemmatize(W[i])
    return(W)                   


#--------------------------------------Storing Positional Index in File------------------------------
def print_positional_index():
    Calc_Inverse_Doc_Freq(totaldocs)
    fileobj = open("Readable_Posting_List.txt", 'w')
    for key, value in sorted(Posting_List.items()):
        fileobj.write(str(key) + " --> " + str(value))
        fileobj.write("\n")
    fileobj.close()
    PL = open('Posting_List.txt', 'w+')
    PL.write(str(Posting_List))
    PL.close()

#-----------This function Reads from doccument call Pre_Processing and create Posting Index-----------
def Make_Indexes():
    docid = 1
    for i in range(totaldocs):
        f = open("ShortStories/"+str(i+1)+'.txt', 'r',encoding="utf8")
        docIdMap[docid] = i
        text=f.read()
        P_words=Pre_Processing(text.lower())
        for word in P_words:
            if not word in Posting_List:
                Posting_List[word]={}
                Posting_List[word][docid]=1
            elif docid not in Posting_List[word]:
                Posting_List[word][docid]=1
            else:
                position=Posting_List[word][docid]
                position=position + 1
                Posting_List[word][docid]=position
        docid = docid + 1
    print_positional_index()

#--------------------Calculating Inverse Doccument Frequency of each Term in a Doccument------------------
def Calc_Inverse_Doc_Freq(N):
    for term in Posting_List:
        if 'Q' in Posting_List[term]:
            T_df=len(Posting_List[term])-1
        else:
            T_df=len(Posting_List[term])
        Posting_List[term]['idf']=math.log10(T_df)/N

#--------------------Calculating TF-IDF Value for each Term with each Doccument------------------
def Calc_Tf_IDf_Value(row_df):
    for column in row_df:
        if(column!='idf' and column!='tf-idf'):
            tfidf = 'tf-idf'+str(column)
            row_df[tfidf] = row_df['idf'] * row_df[column]

#------------------------------------------------------------------------------
def magnitudeQ(dfQ):    #Querry magnitude calculation
    return(np.sqrt( dfQ['tf-idfQ'].dot(dfQ['tf-idfQ']) ))

def magnitudeD(dfD,tf_idf):     #Doccument magnitude calculation 
    return(np.sqrt( dfD[tf_idf].dot(dfD[tf_idf]) ))

def dot_product(df,tf_idf):     #Dotproduct of S(D1,Q)
    return(sum(df[tf_idf]*df['tf-idfQ']))

def Alpha_Ranking(Similarity,i):    #Checking alpha ranking to get similarity
    if(Similarity >= Alpha):
        Result[i]=Similarity
    return Result

#--------------Function to Calculate Cosine Similarity between Doccument and Query using formula------------------
def Cosine_Similarity(df):
    magnitude_Query=magnitudeQ(df)
    i=1
    while(i<=totaldocs):
        tf_idf='tf-idf'+str(i)
        magnitude_Doc = magnitudeD(df,tf_idf)
        Product = dot_product(df,tf_idf)
        Similarity = Product/(magnitude_Doc*magnitude_Query)
        Result = Alpha_Ranking(Similarity,i)
        i=i+1
    return Result

#----------------------Function to add a query vector to a Posting List-----------------------
def AddQueryToPostingList(dict,word,docid):
    if word not in dict:    #if term1 not in posting list, very rare
        dict[word]={}
        dict[word][docid]=1
    elif docid not in dict[word]:   #if 'Q' not in posting list,then add init
        dict[word][docid]=1
    else:                   #if Query term appear multiple times
        vector=dict[word][docid]
        vector=vector + 1
        dict[word][docid]= vector
    return dict

#-----------Function to Trigger every individual process like Calculating TF-IDF,TF-IDFQ,CosinSimilarity---------------
def VSM_Processing(dict):
    files=[]
    filee=1
    for filee in range(totaldocs+1):
        files.append(filee)
    files.append('Q');files.append('idf')
    term=[];term_vector=[]
    for word in dict:
        term_vector.append(dict[word])
        term.append(word)
    vsm_table = pd.DataFrame(term_vector,index=term,columns=files)
    vsm_table = vsm_table.replace(np.nan,0)
    vsm_table.sort_index(inplace=True)
    Calc_Tf_IDf_Value(vsm_table)
    Result=Cosine_Similarity(vsm_table)
    Result=sorted(Result.items(),key=itemgetter(0),reverse=False)
    return Result

#-----------------------------Querry Handling Function----------------------------------------#
def QueryHandler(query):
    PL=open('Posting_List.txt','r')
    dict=eval(PL.read())
    PL.close()
    Parse_query=Pre_Processing(query.lower())
    for term in Parse_query:
        dict=AddQueryToPostingList(dict,term,'Q')
    Result=VSM_Processing(dict)
    return Result

# -------------------------Initiallization Process----------------------------#
def Initiallization():
    Make_Indexes()
    choice=0
    while choice != "2":
        choice = input("\nPress 1 To Enter The Query\nPress 2 To Exit\n")
        print("\n")
        if choice == "1":
            query=input("Search Query : ")
            Result=QueryHandler(query.lower())
            Retrieved_Doccuments=[]
            for res in Result:
                Retrieved_Doccuments.append(res[0])
            print("Length Is : ",len(Result))
            print("Doccuments: ",Retrieved_Doccuments)

def app():
    Initiallization()
app()