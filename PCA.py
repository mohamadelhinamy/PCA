import numpy as np
from numpy import array
from numpy import mean
from numpy.linalg import eig
from numpy import cov
import matplotlib.pyplot as plt
import pandas as pd


def read_input_file(file):
    array2D = []
    f = open(file , 'r')
    for line in f.readlines() :
        line = line.replace('\n','')
        array2D.append(line.split(' '))
    array2D = np.array(array2D).astype(np.float)  
    print(array2D) 
    myArray = np.asarray(array2D)
    print(myArray.shape)

    return myArray 

def calculate_mean(array2D) :
    M = mean(array2D.T, axis=1)
    print('the mean',M)    
    return M

def covariance_matrix(mean,array2D):
    covariance_matrix = np.zeros((len(array2D[0]),len(array2D[0])))
    C = array2D - mean
    print(C)
    for i in C :
        v = np.matrix(i).T*np.matrix(i)
        covariance_matrix  = np.add(covariance_matrix,v) 
    covariance_matrix = covariance_matrix / len(array2D)   
    print('covariance matrix',covariance_matrix)
    covariance_matrix = np.array(covariance_matrix, dtype=float)
    return covariance_matrix
        
   
def PCA(array2D):
    mean = calculate_mean(array2D)
    C = np.subtract(array2D,mean)
    cov1 = covariance_matrix(mean,array2D)
    values, vectors = eig(cov1)
    print('vectors' , vectors)
    print('values', values)
    P = np.dot(vectors.T,array2D.T)
    print('P',P)
    return vectors,values,P

def plot_data(array2D):
    mean = calculate_mean(array2D)
    print(mean[0])
    vectors , values ,P = PCA(array2D)
    print(array2D)
    for i in range(len(P.T)):
        plt.scatter(array2D[i,0],array2D[i,1],c='red')
    plt.quiver(mean[0],mean[1],vectors[0, 0], vectors[1, 0], scale=3, color='blue')
    plt.quiver(mean[0],mean[1],vectors[0, 1], vectors[1, 1],  scale=3, color='green')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('Data_PCA.png')
    plt.show()
def PCA_league_table():
    WS = pd.read_excel('EPL.xlsx')
    WS_np = np.array(WS)
    df = pd.DataFrame(WS_np)
    FTR = df[10]
    df = df.drop([0,1,10],axis=1)
    df=df.values
    df=np.asarray(df)
    mean = calculate_mean(df)
    cov = covariance_matrix(mean,df)
    vectors,values,P = PCA(df)
    print('vecs',vectors)
    print('of zero',vectors[0])
    for idx, vector in enumerate(vectors) :
        home = []
        away = []
        dotto = np.dot(vector.T,df.T)
        for i,value in enumerate(dotto) :
            if FTR[i] == 'H' :
                home.append(value)
            else :
                away.append(value)
        plt.hist([home,away],10,color=['red','blue'],edgecolor='white',linewidth=1,alpha=1)
        d = idx+1
        plt.savefig('Proj_PC'+ str(d) +'.png')       
        plt.show()   
    differ = []
    for vector in vectors :
        home = []
        sum_home = 0
        sum_away = 0
        away = []
        dotto = np.dot(vector.T,df.T)
        for i,value in enumerate(dotto) :
            if FTR[i] == 'H' :
                home.append(value)
                sum_home+=value
            else :
                away.append(value)
                sum_away+=value
        mean_home = sum_home/len(home)
        mean_away = sum_away/len(away)
        diff = abs(mean_home-mean_away) 
        differ.append(diff)
    print(differ)    
    plt.bar(np.arange(len(differ)),differ)
    plt.savefig('Distance.png') 
    plt.show()
             
if __name__ == "__main__":
    array2D = read_input_file('Data.txt')
    plot_data(array2D)
    PCA_league_table()

    
