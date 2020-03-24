#sensitivity analysis on initial opinion distribution
import matplotlib.pyplot as plt
import pylab
import copy
import numpy as np
import seaborn as sns
import pandas as pd
import itertools
from broadcastplatform import *
import multiprocessing as mp
from scipy.stats import uniform
from scipy.stats import randint
from scipy.stats import beta

class sensitivity(object):
    
    def __init__(self,keywords={},times=10):
        self.keywords=keywords#keys:the name of the parameters;values:the range of the sensitivity test
        self.times=times
        self.conversion={'ac':r'$\alpha$','at':r'$d_1$','rc':r'$\beta$','rt':r'$d_2$','ba':r'$b$','bl':r'$l$','bm':r'$m$',
                         'ps':r'$k_P$','ns':r'$k_N$','pi':r'$c$','me':r'$\theta$','ep':r'$p$','sr':r'$x^0_i$','cc':r'$\delta_i$'}
        self.draw_generation()
    
    def draw_generation(self):
        self.draws = []
        for i in range(self.times):
            draw_dict = {}
            draw_dict['beta_draw'] = uniform.rvs(0,1,500)
            draw_dict['bernoulli_draw'] = uniform.rvs(0,1,[500,1000])
            draw_dict['uniform_draw'] = uniform.rvs(0,1,500)
            draw_dict['host_draw'] = uniform.rvs(0,1,1000)
            draw_dict['size_state'] = randint.rvs(0,2000)
            draw_dict['initial_state'] = randint.rvs(0,2000)
            self.draws.append(draw_dict)      

    def initialvalues(self,draw,sr=[-1,1]):
        Size=randint.rvs(0,10,random_state=draw['size_state'])
        Values=uniform.rvs(sr[0],sr[1]-sr[0],size=Size,random_state=draw['initial_state'])
        return Values
    
    def simulation(self,parameters,draw):
        beta_draw = draw['beta_draw']
        bernoulli_draw = draw['bernoulli_draw']
        uniform_draw = draw['uniform_draw']
        host_draw = draw['host_draw']
        tempbp=Platform(beta_draw, bernoulli_draw, uniform_draw, host_draw,
                        ac=parameters['ac'],rc=parameters['rc'],at=parameters['at'],rt=parameters['rt'],ba=parameters['ba'],
                        bm=parameters['bm'],bl=parameters['bl'],ps=parameters['ps'],ns=parameters['ns'],pi=parameters['pi'],
                        me=parameters['me'],ep=parameters['ep'],sr=parameters['sr'],p_mode=1, host_mode='triang',te=1000,p=500,
                        beta_a=parameters['cc'][0], beta_b=parameters['cc'][1]
                       )
        #print(parameters)
        tempbp.Evolution(InitialValues=self.initialvalues(draw, sr=parameters['sr']))
        return {"AverageOpinion":tempbp.AveSeries(show=False)[999],
                "AverageHost":tempbp.AveHost(),
                "FinalRatio":tempbp.FinalRatio(show=False),
                "ConversionRatio":tempbp.ConversionRatio()
                }

    def replication(self,parameters):
        #pool=mp.Pool()
        result=list(itertools.starmap(self.simulation,[(parameters,draw) for draw in self.draws]))
        temp=[]
        for i in result:
            temp+=i["FinalRatio"]
        hist=np.histogram(temp,bins=20,range=(-1.0,1.0))
        return {"AverageOpinion":np.mean([x["AverageOpinion"] for x in result]),
                "AverageHost":np.mean([x["AverageHost"] for x in result]),
                "FinalRatio":hist,
                "ConversionRatio":np.mean([x["ConversionRatio"] for x in result],axis=0),
                "parameter":parameters
                }

    #sensitive analysis on all the parameters listed in the keywords, respectively 
    def singletest(self,parameters):
        pool=mp.Pool()
        parameterpool={}
        result={}
        
        #setting the parameterpool
        for k in self.keywords.keys():
            parameterpool[k]=[]
            for i in self.keywords[k]:
                temp={'ac':parameters['ac'],'rc':parameters['rc'],'at':parameters['at'],
                      'rt':parameters['rt'],'ba':parameters['ba'],'bl':parameters['bl'],
                      'bm':parameters['bm'],'ps':parameters['ps'],'ns':parameters['ns'],
                      'pi':parameters['pi'],'me':parameters['me'],'ep':parameters['ep'],
                      'sr':parameters['sr'],'cc':parameters['cc']
                      }
                temp[k]=i
                parameterpool[k].append(temp)
        #sensitive analysis
        for k in self.keywords.keys():
            result[k]=pool.map(self.replication,parameterpool[k])
            
        return result#result is a dictionary
    
    def multipletest(self,parameters):
        pool=mp.Pool()
        parameterpool=[]
        
        for i in range(0,len(list(self.keywords.values())[0])):
            temp={'ac':parameters['ac'],'rc':parameters['rc'],'at':parameters['at'],
                  'rt':parameters['rt'],'ba':parameters['ba'],'bl':parameters['bl'],
                  'bm':parameters['bm'],'ps':parameters['ps'],'ns':parameters['ns'],
                  'pi':parameters['pi'],'me':parameters['me'],'ep':parameters['ep'],
                  'sr':parameters['sr'],'cc':parameters['cc']
                  }
            for k in self.keywords.keys():
                temp[k]=self.keywords[k][i]
            parameterpool.append(temp)
            
        result=pool.map(self.replication,parameterpool)
        
        return result#result is a list
        
    def singlevisualize(self,result):
        pylab.rcParams['figure.figsize'] = (30.0, 20.0)
        for k in result.keys():
            avo=plt.subplot(131)
            fr=plt.subplot(132)
            con=plt.subplot(133)
           
            avo.plot(self.keywords[k],[x['AverageOpinion'] for x in result[k]],label='AverageOpinion',marker='o')
            avo.plot(self.keywords[k],[x['parameter']['bm'] for x in result[k]],label='host-m',marker='1')
            avo.plot(self.keywords[k],[x['parameter']['bl'] for x in result[k]],label='host-l',marker='2')
            avo.plot(self.keywords[k],[x['AverageHost'] for x in result[k]],label='host-ave',marker='3')
            
            con.plot(self.keywords[k],[x['ConversionRatio'][0] for x in result[k]],label='PosToNeg',marker='2')
            con.plot(self.keywords[k],[x['ConversionRatio'][1] for x in result[k]],label='NegToPos',marker='1')
            
            for i in result[k]:
                label=self.conversion[k]+'='+str(round(i['parameter'][k],2))
                xaxis=[(i['FinalRatio'][1][x]+i['FinalRatio'][1][x+1])/2 for x in range(0,20)]
                yaxis=[y/(500*self.times) for y in i['FinalRatio'][0]]
                fr.plot(xaxis,yaxis,label=label,marker=randint.rvs(0,10))
                #tempdf=pd.DataFrame({legend:i['FinalRatio']})
                #sns.kdeplot(tempdf[legend],ax=fr,bw=0.05)
                
            avo.set_xlim([min(self.keywords[k]),max(self.keywords[k])])
            avo.set_ylim([-1,1])
            fr.set_xlim([-1,1])
            #fr.set_ylim([0,1])
            con.set_xlim([min(self.keywords[k]),max(self.keywords[k])])
            con.set_ylim([0,1])
            
            avo.set_xlabel(self.conversion[k])
            avo.set_ylabel('Average Opinion')
            fr.set_xlabel('Opinion Value')
            fr.set_ylabel('Final Ratio')
            con.set_xlabel(self.conversion[k])
            con.set_ylabel('Conversion Ratio')
            
            avo.legend()
            fr.legend()
            con.legend()
            
            plt.show()
            

            result[k]=pool.map(self.replication,parameterpool[k])
            
        return result#result is a dictionary
    
    def multipletest(self,parameters):
        pool=mp.Pool()
        parameterpool=[]
        
        for i in range(0,len(list(self.keywords.values())[0])):
            temp={'ac':parameters['ac'],'rc':parameters['rc'],'at':parameters['at'],
                  'rt':parameters['rt'],'ba':parameters['ba'],'bl':parameters['bl'],
                  'bm':parameters['bm'],'ps':parameters['ps'],'ns':parameters['ns'],
                  'pi':parameters['pi'],'me':parameters['me'],'ep':parameters['ep'],
                  'sr':parameters['sr'],'cc':parameters['cc'],'bt':parameters['bt']
                  }
            for k in self.keywords.keys():
                temp[k]=self.keywords[k][i]
            parameterpool.append(temp)
            
        result=pool.map(self.replication,parameterpool)
        
        return result#result is a list


                  'pi':parameters['pi'],'me':parameters['me'],'ep':parameters['ep'],
                  'sr':parameters['sr'],'cc':parameters['cc'],'bd':parameters['bd'],'bf':parameters['bf']
                  }
            for k in self.keywords.keys():
                temp[k]=self.keywords[k][i]
            parameterpool.append(temp)
            
        result=pool.map(self.replication,parameterpool)