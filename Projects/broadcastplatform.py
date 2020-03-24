import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import copy
import random
import pandas as pd
from scipy.stats import *
import pylab

class Platform(object):
    
    def __init__(self, beta_draw, bernoulli_draw, uniform_draw, host_draw,at=1.0,rt=1.0,ba=0.5,bm=0.75,bl=0.0,ps=0.5,ns=1,pi=0.25,
                 ac=0.5,rc=0.5,me=0.5,ep=0.5,sr=[-1,1],beta_a=1, beta_b=100, p_mode=1, host_mode='triang',te=150,p=500):
        #the parameters and coefficients
        self.ac=ac#assimilation coefficient
        self.rc=rc#repulsion coefficient
        self.at=at#assimilation threshold
        self.rt=rt#repulsion threshold
        self.ba=ba#broadcast attention ratio
        self.bm=bm#mean of the information source value
        self.bl=bl#the lowest information source value produced by the host
        self.ps=ps#positive slope
        self.ns=ns#negative slope
        self.pi=pi#positive intercept
        self.me=me#marginal exponent
        self.epma=ep#expression possiblity-majority
        self.epmi=1-ep#expression possiblity-minority
        self.beta_a=beta_a
        self.beta_b=beta_b
        
        self.beta_draw = beta_draw
        self.bernoulli_draw = bernoulli_draw
        self.uniform_draw = uniform_draw
        self.host_draw =  host_draw
        
        self.te=te#end point
        self.sr=sr#initial range
        self.p=p#number of audiences
        self.p_mode=p_mode#the mode of opinion perceiption
        self.host_mode=host_mode#the distribution of the host mode 
        
        self.ec=list()
        self.A=list()
        self.G = nx.complete_graph(self.p+1) #include 500 audiences and 1 host in a complete graph
        nx.set_node_attributes(self.G,None, 'SelfOpinion') 
        nx.set_node_attributes(self.G,None, 'OthersOpinion')  
        nx.set_node_attributes(self.G,None, 'ExpressionState')
        nx.set_node_attributes(self.G,None, 'ExpressionMajority')
        nx.set_node_attributes(self.G,None, 'ExpressionMinority')
    
    #construct the transition function
    def AudienceOpinionTransition(self,n):
        node=self.G.nodes[n]
        nextOpinion=0
        so=node['SelfOpinion']
        oo=node['OthersOpinion']
        difference=oo-so
        if abs(difference)<self.at:
            nextOpinion=so+0.5*self.ac*(1-abs(so))*difference
        elif abs(difference)<=self.rt:
            nextOpinion=so
        else:
            nextOpinion=so-0.5*self.rc*(1-abs(so))*difference
        return nextOpinion

    def HostOpinionTransition(self,n):
        node=self.G.nodes[n]
        nextOpinion=0 
        so=node['SelfOpinion']
        oo=node['OthersOpinion']
        difference=oo-so
        if abs(difference)<self.at/self.scn:
            nextOpinion=so+self.scm*0.5*self.ac*(1-abs(so))*difference
        elif abs(difference)<=self.rt*self.scn:
            nextOpinion=so
        else:
            nextOpinion=so-self.scm*0.5*self.rc*(1-abs(so))*difference
        return nextOpinion
    
    #construct the expression mechanism
    def ExpressionMode(self,n,draw):
        node=self.G.nodes[n]
        #conformity=pow(self.ep,pow(len(self.A)-1,1))
        #inconformity=pow(1-self.ep,pow(len(self.A)-1,1))
        if node['SelfOpinion']*node['OthersOpinion']>0:#abs(node['SelfOpinion']-node['OthersOpinion'])<=self.at
            self.G.nodes[n]['ExpressionState']=bernoulli.ppf(draw, node['ExpressionMajority'])
        elif node['SelfOpinion']*node['OthersOpinion']<0:# abs(node['SelfOpinion']-node['OthersOpinion'])>=self.rt
            self.G.nodes[n]['ExpressionState']=bernoulli.ppf(draw, node['ExpressionMinority'])
        else:
            self.G.nodes[n]['ExpressionState']=0
    
    def UpdateA(self):
        self.A.clear()
        for i in np.arange(0,self.p+1,1):
            if self.G.nodes[i]['ExpressionState']==1:
                self.A.append(i)
        return self.A
    
    #construct the cognitive bias
    def BiasedOpinion(self,opinion):
        if self.p_mode == 1:
            if opinion>0:
                biasedOpinion=pow(opinion,self.me)*self.ps+self.pi
            else:
                biasedOpinion=pow((-1*opinion),self.me)*self.ns*(-1)
        else:
            if opinion>0:
                biasedOpinion=(opinion*self.ps+self.pi)**self.me
            else:
                biasedOpinion=-(-1*opinion*self.ns)**self.me
        return biasedOpinion
    
    #construct the integration of others'opinion
    def IntegratedOpinion(self,i): 
        integratedOpinion=0
        for j in self.A:
            if j!=i:
                integratedOpinion+=(self.AttentionWeight(i,j)*self.BiasedOpinion(self.G.nodes[j]['SelfOpinion']))
            else:
                continue
        return integratedOpinion
    
    #construct the attention weight
    def AttentionWeight(self,i,j):
        #expressed=pow(self.ba,pow(len(self.A)-2,0.5))
        #unexpressed=pow(self.ba,pow(len(self.A)-1,0.5))
        if i==0:
            attentionWeight=1/(len(self.A)-1) if len(self.A)>1 else 0
        elif j==0 and self.G.nodes[i]['ExpressionState']==1:
            attentionWeight=pow(self.ba,pow(len(self.A)-2,0.5))
        elif j==0 and self.G.nodes[i]['ExpressionState']==0:
            attentionWeight=pow(self.ba,pow(len(self.A)-1,0.5))
        elif j!=0 and self.G.nodes[i]['ExpressionState']==1:
            attentionWeight=(1-pow(self.ba,pow(len(self.A)-2,0.5)))/(len(self.A)-2)
        else:
            attentionWeight=(1-pow(self.ba,pow(len(self.A)-1,0.5)))/(len(self.A)-1)
        return attentionWeight
    
    #def ExpressionInitialization(self,RealDensity):
    #    keys=list(RealDensity.keys())
    #    random.shuffle(keys)
    #    temp=1
    #    for i in keys:
    #        for j in range(temp,temp+RealDensity[i]):
    #            self.G.nodes[j]['ExpressionMajority']=2*(i/self.te)*(self.epma/(self.epma+self.epmi))
    #            self.G.nodes[j]['ExpressionMinority']=2*(i/self.te)*(self.epmi/(self.epma+self.epmi))
    #        temp=temp+RealDensity[i]
            
    #initialize and store the opinion distribution at the beginning
    def Initialization(self,Values=[]):
        self.ec.clear()
        temp=list()
        #self.ExpressionInitialization(RealDensity)
        nx.set_node_attributes(self.G, 0, 'Time')  
        self.G.nodes[0]['SelfOpinion']=triang.ppf(self.host_draw[0], c=(self.bm-self.bl)/(1-self.bl),loc=self.bl,scale=1-self.bl)\
            if self.host_mode=='triang' else min(max(norm.ppf(self.host_draw[0], self.bm, self.bl),0),1)
        #set the node 0 to be the host with initial information value 0.5, variance value 0-norm.rvs(self.bm,self.bv)
        self.G.nodes[0]['ExpressionState']=1 #set the initial expression state of the host to be always active
        for i in np.arange(1,self.p+1,1):
            self.G.nodes[i]['SelfOpinion']=uniform.ppf(self.uniform_draw[i-1],self.sr[0],self.sr[1]-self.sr[0])#0
            self.G.nodes[i]['ExpressionState']=0
            self.G.nodes[i]['ExpressionMajority']=\
            beta.ppf(self.beta_draw[i-1],self.beta_a,self.beta_b,loc=2*0.7/self.te)*(self.epma/(self.epma+self.epmi))
            self.G.nodes[i]['ExpressionMinority']=\
            beta.ppf(self.beta_draw[i-1],self.beta_a,self.beta_b,loc=2*0.7/self.te)*(self.epmi/(self.epma+self.epmi))
        #Setting the initial opinion values of those expressing agents
        for k in np.arange(1,len(Values)+1,1):
            self.G.nodes[k]['SelfOpinion']=Values[k-1]
            self.G.nodes[k]['ExpressionState']= 1
        self.UpdateA()
        for i in np.arange(0,self.p+1,1):
            self.G.nodes[i]['OthersOpinion']=self.IntegratedOpinion(i)
            #node=copy.deepcopy(self.G.nodes[i])
            nodeinfo={'Index':i,'SelfOpinion':self.G.nodes[i]['SelfOpinion'],
                      'ExpressionState':self.G.nodes[i]['ExpressionState'],'Time':self.G.nodes[i]['Time']}
            temp.append(nodeinfo)
        self.ec.append(temp)        
        
    #opinion evolution
    def Evolution(self,InitialValues=[]):#(self.epma*self.te):self.p
        self.Initialization(InitialValues)
        for t in np.arange(1,self.te,1):
            temp=list()
            nx.set_node_attributes(self.G,t, 'Time')
            self.G.nodes[0]['SelfOpinion']= triang.ppf(self.host_draw[t], c=(self.bm-self.bl)/(1-self.bl),loc=self.bl,scale=1-self.bl)\
               if self.host_mode=='triang' else min(max(norm.ppf(self.host_draw[t], self.bm, self.bl),0),1)
            for i in np.arange(1,self.p+1,1):
                self.G.nodes[i]['SelfOpinion']=self.AudienceOpinionTransition(i)
                self.ExpressionMode(i,self.bernoulli_draw[i-1,t-1])
            self.UpdateA()
            for i in np.arange(0,self.p+1,1):
                self.G.nodes[i]['OthersOpinion']=self.IntegratedOpinion(i)
                nodeinfo={'Index':i,'SelfOpinion':self.G.nodes[i]['SelfOpinion'],
                          'ExpressionState':self.G.nodes[i]['ExpressionState'],'Time':self.G.nodes[i]['Time']}
                temp.append(nodeinfo)
            self.ec.append(temp)
        
    def ExpressedSeries(self):
        if self.ec==[]:
            return False
        Series=[]
        for x in self.ec:
            temp=[]
            for i in x[1:]:
                if i['ExpressionState']==1:
                    temp.append(i)
            Series.append(temp)
        return Series
                    
    def PlotAll(self):
        if self.ec==None:
            return False
        tz=np.arange(0,self.te,1)
        plt.plot(tz,list(x[0]['SelfOpinion'] for x in self.ec),'r')
        for i in np.arange(1,self.p+1,1):
            plt.plot(tz,list(x[i]['SelfOpinion'] for x in self.ec),'b')
        plt.xlabel('time')
        plt.ylabel('opinion value')
        plt.title('Pic. Opinion Evolution (all nodes)')
        plt.legend(('host','audience'), loc='lower right')
        plt.xlim([0,self.te])
        #plt.ylim([-1.0,1.0])
        plt.show()
    
    def AveSeries(self,show=True):
        if self.ec==None:
            return False
        tz=np.arange(0,self.te,1)
        avc=[]
        for x in self.ec:
            average=np.mean(list(y['SelfOpinion'] for y in x[1:]))
            avc.append(average)
        if show==True:
            plt.plot(tz,avc,label='Simulated')
            plt.xlabel('time')
            plt.ylabel('opinion value')
            plt.title('Average All Opinion Evolutionary Data')
            plt.xlim([0,self.te])
            plt.ylim([-1.0,1.0])
            #plt.savefig('sensitivity pictures/'+'AveSeries')
            plt.show()
        return avc
    
    def AveExpressedSeries(self,show=True):
        Series=self.ExpressedSeries()
        tz=np.arange(0,self.te,1)
        aeoc=[]
        aevar=[]
        for x in Series:
            if len(x)==0:
                aeoc.append(0)
                aevar.append(0)
            else:
                aeoc.append(np.mean([i['SelfOpinion'] for i in x]))
                aevar.append(np.var([i['SelfOpinion'] for i in x]))
        if show==True:
            plt.plot(tz,aeoc,label='Simulated')
            plt.xlabel('time')
            plt.ylabel('opinion value')
            plt.title('Average Expressed Opinion Evolutionary Data')
            plt.xlim([0,self.te])
            plt.ylim([-1.0,1.0])
            #plt.savefig('sensitivity pictures/'+'AveExpressedSeries')
            plt.show()
        return (aeoc,aevar)
    
    def AveHost(self):
        if self.ec==[]:
            return False
        averagehost=np.mean([x[0]['SelfOpinion'] for x in self.ec])
        return averagehost
    
    def PosRatioSeries(self,show=True):
        Series=self.ExpressedSeries()
        tz=np.arange(0,self.te,1)
        prc=[]
        for x in Series:
            PosNumber=0
            NegNumber=0
            if len(x)==0:
                prc.append(0.5)
            else:
                for i in x:
                    if i['SelfOpinion']>=0:
                        PosNumber+=1
                    if i['SelfOpinion']<=0:
                        NegNumber+=1
                prc.append(PosNumber/(PosNumber+NegNumber))
        if show==True:
            plt.plot(tz,prc,label='Simulated')
            plt.xlabel('Time')
            plt.ylabel('Positivity Ratio')
            plt.title('Positive Opinion Ratio Series')
            plt.xlim([0,self.te])
            plt.ylim([0.0,1.0])
            #plt.savefig('sensitivity pictures/'+'PosRatioSeries')
            plt.show()
        return prc
    
    def ExpressionRatioSeries(self,show=True):
        Series=self.ExpressedSeries()
        tz=np.arange(0,self.te,1)
        ers=[]
        for x in Series:
            ers.append(len(x)/self.p)
        if show==True:
            pylab.rcParams['figure.figsize'] = (40.0, 8.0)
            plt.plot(tz,ers,label='Simulated')
            plt.xlabel('Time')
            plt.ylabel('Expression Ratio')
            plt.title('Expression Ratio Series-Simulated')
            plt.xlim([0,self.te])
            #plt.savefig('sensitivity pictures/'+'ExpressionRatioSeries')
            plt.show() 
        return ers
    
    def ExpressionDensity(self,show=True):
        Series=self.ExpressedSeries()
        ed=pd.DataFrame(data=[(x,0) for x in range(1,self.p+1)],columns=['index','frequency'])
        for i in Series:
            for j in i:
                ed.iloc[j['Index']-1]['frequency']+=1
        Density=ed['index'].groupby(ed['frequency']).count()
        Density=Density.sort_values(ascending=False)
        if show==True:
            pylab.rcParams['figure.figsize'] = (15.0, 8.0)
            plt.plot([x for x in Density.values],[y for y in Density.index],label='Simulated')
            plt.xlabel('Count of Audiences')
            plt.ylabel('Expression Frequency')
            plt.title('Expression Density-Simulated')
            plt.show()
        return Density
    
    def FinalRatio(self,show=True):
        if self.ec==[]:
            return False
        result=[x['SelfOpinion'] for x in self.ec[self.te-1][1:]]
        if show==True:
            ax = sns.kdeplot(result)
            plt.show()
        return result
    
    def ConversionRatio(self):
        if self.ec==[]:
            return False
        PosToNeg=0
        NegToPos=0
        for x in range(1,self.p+1):
            start=self.ec[0][x]['SelfOpinion']
            result=self.ec[self.te-1][x]['SelfOpinion']
            if start*result<0 and start>0:
                PosToNeg+=1
            elif start*result<0 and start<0:
                NegToPos+=1
        
        return (PosToNeg/self.p,NegToPos/self.p)