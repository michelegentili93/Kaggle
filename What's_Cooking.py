
import json
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import numpy as np
import pandas as pd
import time

time0=time.time()
##step 0

risposta=raw_input('what kind of Precision do you want: low (5 min, no correlation) medium (30 min) high(2:30 h, best score in kaggle) ?\n')

vuoi_correlazioni=True
if risposta in ['low','l','1']:
    iterazioni=15
    n_min_nnz=70
    salti=100
    da_analizzare=20
    vuoi_correlazioni=False

if risposta in ['medium','m','2']:
    iterazioni=30
    n_min_nnz=40
    salti=15
    da_analizzare=75
    
if risposta in ['high','h','3']:
    iterazioni=250
    n_min_nnz=30
    salti=20
    da_analizzare=100
    

def ordered_intersection(lista_1,lista_2):
    punt_1=0
    punt_2=0
    lista=[]
    lista_1=sorted(lista_1)
    lista_2=sorted(lista_2)
    while True:
        if punt_1==len(lista_1) or punt_2==len(lista_2):
            break
        if lista_1[punt_1]==lista_2[punt_2]:
            lista.append(lista_1[punt_1])
            punt_1+=1
            punt_2+=1
            continue
        if lista_2[punt_2]<lista_1[punt_1]:
            punt_2+=1
            continue
        punt_1+=1
    return lista
        

def h(teta,x):
    return (1.0/(1+(np.exp(-x.dot(teta)))))
def f(teta,x,y):
    return sum(y*np.log(h(teta,x))+(1-y)*np.log(1-h(teta,x)))    
def grad(teta,x,y):
    return (x.transpose().dot(y-h(teta,x)))
def giusti(teta,x,y):
    return sum((h(teta,x)>0.5)==y)*1.0/len(y)
def LogReg(x,y):
    
    teta=np.zeros(x.shape[1])
    err=100
    alpha=0.01
    a=2
    j=0
    epsilon=10**(-5)
    while True:
        #aggiorno stato a partire da teta
        alphai=alpha
        found=False
        
        tetanew=teta.copy()
        #esplora direzioni congiunte
        for t in range(1,5):
            tetanew=teta+alphai*grad(teta,x,y)
            if f(tetanew,x,y)>=f(teta,x,y):
                break
            else:
                alphai=alpha/(10.0**t)
                continue
        teta=tetanew.copy()
        j+=1 
        if (j>15 and giusti(teta,x,y)>0.995) or j>iterazioni:
            break
    return teta,j

##     Step 1)
print '\nStep 1) store data\n'
time1=time.time()
with open('train.json') as data_file:    
    data = json.load(data_file)
    
i=1
ingredients={}
decript_ingredients={}

j=1
nations={}
decript_nations={}

k=0
ID={}

for kitchen in data:
    ID[kitchen['id']]=k
    for ingr in kitchen['ingredients']:
        if ingr not in ingredients:
            ingredients[ingr]=i
            decript_ingredients[i]=ingr
            i+=1
    if kitchen['cuisine'] not in nations:
        nations[kitchen['cuisine']]=j
        decript_nations[j]=kitchen['cuisine']
        j+=1
    k+=1
    
    
x_training=dok_matrix((len(data),len(ingredients)+1))
x_training.update({(ID[ricetta['id']],ingredients[ingr]):1  for ricetta in data for ingr in ricetta['ingredients']})
x_training.update({(i,0):1 for i in range(x_training.shape[0])})
x_training=x_training.tocsr()
y_training=dok_matrix((len(data),1))
y_training.update({(ID[kit['id']],0) : nations[kit['cuisine']] for kit in data})
y_training=y_training.tocsr()
sparse_nazionalita=y_training.copy()

print time.time()-time1
##      Step 2)
if vuoi_correlazioni:

    print '\nStep 2)  \n'

    time1=time.time()
    k=1

    correlazioni={}
    correlazioni1={}
    relazioni={}

    print 'Seeking joint ingredients in: '
    for naz in decript_nations:
        print decript_nations[naz],
        x_sel=x_training[(y_training==naz).toarray()[:,0]]
        n_row=x_sel.shape[0]
        n_col=x_sel.shape[1]
        ricette_nazione=x_sel.shape[0]
        for div in range(n_min_nnz,100,salti):
            to_check=set()
            for j in xrange(1,n_col):
                nnz=x_sel[:,j].nnz

                if nnz>div:
                    to_check.add(j)
            if len(to_check)<=da_analizzare:
                break

        updated_set=to_check.copy()
        for i in to_check:
            updated_set.discard(i)
            for j in updated_set:
                lista1=x_sel[:,i].nonzero()[0]
                lista2=x_sel[:,j].nonzero()[0]
                intersezione=ordered_intersection(lista1,lista2)
                len_intersezione= len(intersezione)
                if len_intersezione>x_sel.shape[0]*1.0/10:
                    correlazioni[k]=(i,j)
                    k+=1
                    if len_intersezione>10:
                        try:
                            relazioni[i].append(j)
                        except: relazioni[i]=[j]

        updated_set1=updated_set.copy()
        coppie=set(correlazioni.values())
        for (i,j) in coppie:
            try:
                to_check_ij=ordered_intersection(relazioni[i],relazioni[j])
            except: continue
            for m in to_check_ij:
                lista1=x_sel[:,m].nonzero()[0]
                lista2=x_sel[:,m].nonzero()[0]
                lista3=x_sel[:,m].nonzero()[0]  
                intersezione1=ordered_intersection(ordered_intersection(lista1,lista2),lista3)
                len_intersezione1=len(intersezione1)
                if len_intersezione1>x_sel.shape[0]*1.0/100:
                    correlazioni1[k]=(i,j,m)

        print len(correlazioni), ' joint ingredients'

    print time.time()-time1
    ##     Step 3)

    print '\nStep 3) updating matrix\n'

    time1=time.time()
    x_data_corr= dok_matrix(x_training)
    x_data_corr._shape=((x_data_corr.shape[0],x_training.shape[1]+len(correlazioni)))



    for (col,(i,j)) in correlazioni.items():
        res=ordered_intersection(x_data_corr[:,(i+1)].nonzero()[0],x_data_corr[:,(j+1)].nonzero()[0])
        for righe in res:
            x_data_corr.update({(righe,col+x_training.shape[1]-2):1 })
    for (col,(i,j,m)) in correlazioni1.items():
        res=ordered_intersection(x_data_corr[:,m].nonzero()[0],ordered_intersection(x_data_corr[:,i].nonzero()[0],x_data_corr[:,j].nonzero()[0]))
        for righe in res:
            x_data_corr.update({(righe,col+x_training.shape[1]-2):1 })
    x_data_corr=x_data_corr.tocsr()

if not vuoi_correlazioni:
    x_data_corr=x_training
with open('test.json') as data_file:    
    data_test = json.load(data_file)
k=0  

ID={}
decript_ID={}
list_ID_ingredienti=[]
for kitchen in data_test:
    ID[kitchen['id']]=k
    decript_ID[k]=kitchen['id']
    for ingr in kitchen['ingredients']:
        if ingr in ingredients:
            list_ID_ingredienti.append([k,ingredients[ingr]])
    k+=1

x_test=dok_matrix((len(data_test),x_data_corr.shape[1]))
x_test.update({(Id,ricetta):1 for (Id,ricetta) in list_ID_ingredienti})
x_test.update({(i,0):1 for i in range(x_test.shape[0])})
if vuoi_correlazioni:
    for (col,(i,j)) in correlazioni.items():
        res=set(x_test[:,i].nonzero()[0]).intersection(set(x_test[:,j].nonzero()[0]))
        for righe in res:
            x_test.update({(righe,col+x_training.shape[1]-2):1 })
    for (col,(i,j,m)) in correlazioni1.items():
        res=ordered_intersection(x_test[:,m].nonzero()[0],ordered_intersection(x_test[:,i].nonzero()[0],x_test[:,j].nonzero()[0]))
        for righe in res:
            x_test.update({(righe,col+x_training.shape[1]-2):1 })
x_test=x_test.tocsr()


print time.time()-time1    
##     Step 4)

print '\nStep 4) \n'
time1=time.time()

dataframe=pd.DataFrame()
for i in range(1,len(nations)+1):
    print 'Computing h(x) for ' , decript_nations[i]
    y_training=sparse_nazionalita
    y_training=(y_training==i).toarray()[:,0]
    teta,j=LogReg(x_data_corr,y_training)
    dataframe[decript_nations[i]]=h(teta[abs(teta)>0.01],x_test[:,abs(teta)>0.01])
    
    
y_training=sparse_nazionalita
rowmax = dataframe.max(axis=1)
max_h=(np.where(dataframe.values == rowmax[:,None])[1]+1)

print time.time()-time1
##   Step 5)
print '\nStep 5) to csv\n'
time1=time.time()
lista_submit=[]
for i in range(len(max_h)):
    lista_submit.append([decript_ID[i],decript_nations[max_h[i]]])
dataframe_submit=pd.DataFrame(lista_submit,columns=['ID','cuisine'])
dataframe_submit.to_csv('1462272.csv', index=False)

print time.time()-time1
print 'time: ' , time.time()-time0
