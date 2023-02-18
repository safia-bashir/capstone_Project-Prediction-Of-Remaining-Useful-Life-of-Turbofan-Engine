
import pandas as pd
import io

from fastapi import FastAPI, UploadFile,File,Form
#from fastapi import FastAPI
from cnn_svr import Cnn_Svr,Cnn_Svr2
#from test import Cnn_Svr2
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

class User_input(BaseModel):
    en: int
    cycle: int



columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
            "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
            ,"sensor20","sensor21"]


    #test_data =pd.read_csv("../data/test_FD001.txt", sep= "\s+", header = None,names=columns)
    #test_data =pd.read_csv("../data/test_FD001.txt", sep= "\s+", header = None) 
    #test_data =pd.read_csv(test_cv, sep= "\s+", header = None,names=columns)# lets see how to do this 

    
#tes = pd.read_csv( r"/mnt/c/Users/safsa/Desktop/Front_back_end/data/test_FD001.txt", sep = "\s+",header = None,names=columns)

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(input:User_input):
    
 
    result = Cnn_Svr(input.en,input.cycle)
  
    result = int(result)
    
    
    return result
@app.post("/m")
def pre(input:User_input):
    
 
    m= Cnn_Svr2(input.en,input.cycle)
  
    m = int(m)
    
    return m      
  