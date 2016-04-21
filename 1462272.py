#michele gentili, 1462272
try:
    import csv
    import numpy as np
    import pandas

    data = pandas.read_csv(
        "training.csv",
        na_values = np.nan,
        header    = True,
        usecols   = [1, 2],
        names     = ['SAT', 'GPA'],
    )
    x1=np.array(data['SAT'])
    y1=np.array(data['GPA'])

    def ott(tetainput):

        #normalizzo
        x=(x1-np.mean(x1))/np.sqrt(np.var(x1))
        y=(y1-np.mean(y1))/np.sqrt(np.var(y1))

        #definisco funzione da ottimizzare e calcolo gradiente tutto in funzione di teta
        def f(teta):
            return 0.5*sum((teta[0]+teta[1]*x-y)**2)
        def grad0(teta):
            return np.dot(teta[0]+teta[1]*x-y,np.ones(len(x)))
        def grad1(teta):
            return np.dot(teta[0]+teta[1]*x-y,x)
        def gradiente(teta):
            return np.array([grad0(teta),grad1(teta)])

        #calcolo teta di partenza come array, assicurandomi che sia un float

        teta=np.array([float(tetainput[0]),float(tetainput[1])])

        #definisco lunghezza del primo passo
        alpha=1

        #contatore per stampare ( ne stampo 10 ogni 1000 iterazioni) e sapere in che iterazione sto
        a=0
        j=0

        #definisco lo spostamento minimo per cui continuo
        epsilon=10**(-8)

        #start

        while True:

            #aggiorno stato a partire da teta ( gradiente del punto, alphai e' quello che cambia, found se trova il punto, tetanew e' il teta di prova

            grad=gradiente(teta)
            alphai=alpha
            found=False
            tetanew=teta.copy()

            #esplora entrambe le direzioni contemporaneamente

            for t in range(20):
                tetanew=teta-alphai*grad
                tetanew2=teta-10*alphai*grad
                if f(tetanew)<=f(teta) and f(teta)<=f(tetanew2):
                    found=True
                    break
                elif f(teta)>f(tetanew2) :
                    alphai=alpha*10**t
                    continue
                elif f(teta)<f(tetanew):
                    alphai=alpha/(10.0**t)
                    continue

            #esplora solo lungo teta0

            if sum(abs(teta-tetanew))<epsilon or not found:
                found=False
                tetanew=teta.copy()
                alphai=alpha
                for t in range(40):
                    tetanew[0]=teta[0]-alphai*grad[0]
                    tetanew2[0]=teta[0]-5*alphai*grad[0]
                    if f(tetanew)<=f(teta) and f(teta)<=f(tetanew2):
                        found=True
                        break
                    elif f(teta)>=f(tetanew2) :                
                        alphai=alpha*5**t
                        continue
                    elif f(teta)<=f(tetanew):
                        alphai=alpha/(5.0**t)
                        continue

                #esplora solo lungo teta1

            if not found or sum(abs((teta-tetanew)))<epsilon:
                tetanew=teta.copy()
                alphai=alpha
                for t in range(40):
                    tetanew[1]=teta[1]-alphai*grad[1]
                    tetanew2[1]=teta[1]-5*alphai*grad[1]
                    if f(tetanew)<=f(teta) and f(teta)<=f(tetanew2):
                        found=True
                        break
                    elif f(teta)>f(tetanew2) :
                        alphai=alpha*5**t
                        continue
                    elif f(teta)<=f(tetanew):
                        alphai=alpha/(5.0**t)
                        continue

            #condizione di uscita

            if not found or sum(abs((teta-tetanew)))<epsilon:
                break

            #se non esce, aggiorna il teta
            teta=tetanew.copy()

            #opzioni stampa
            a-=1 
            j+=1
            if a<-1000:
                a=10
            if  a>0 or j<10:
                    print j,'teta [' , round(teta[0],3),round(teta[1],3),'] grad=[', round(grad[0]),round(grad[1]),'] err: ',round(f(teta))

        #normalizzo i teta sui valori iniziali
        teta[1]=teta[1]*(np.sqrt(np.var(y1))/np.sqrt(np.var(x1)))
        teta[0]=np.mean(y1)-teta[1]*np.mean(x1)

        print round(teta[0],3), round(teta[1],5)

        return teta

    #scrivo la stima del SAT sul nuovo file

    data = pandas.read_csv(
        "testfile.csv",
        na_values = np.nan,
        header    = False,
        usecols   = [0,1],
        names     = ['ID','GPA'],
    )
    id=data['ID']
    xtest=np.array(data['GPA'])

    teta0=-100000
    teta1=+100000
    teta=[teta0,teta1]

    ytest=np.dot(ott(teta),[np.ones(len(xtest)),xtest])

    dat=pandas.DataFrame(ytest,index=data['ID'],columns=['GPA'])
    dat.to_csv('1462272.csv')
    print '\n y-test'
    print dat[:10]
    print '...'
except: print '\nhi prof: \n\ntestfile.csv and training.csv needed in the same directory of this .py \nthank you!\n'
