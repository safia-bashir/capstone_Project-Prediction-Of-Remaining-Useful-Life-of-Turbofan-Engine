import streamlit as st 
import io
import json
import requests
import pandas as pd 


text = "Accurately predict the remaining Usful life of your turbofan engines with our\nmachine learning-based web app.Monitor engine health and plan maintenance with ease."
col1, col2 = st.columns([1, 1])
with col1:
    st.title('Prediction of Remaining Useful Life (RUL) of turbofan engine')
with col2:
    st.image("RUL.png") 
st.subheader("Accurately predict the remaining useful life of your turbofan engines with our machine learning-based web app.")
st.text("To predict the remaining useful life of your turbofan engines,\nyou need to follow the steps below:") 
st.markdown(""" 
1. Click on the "Upload" button on the left sidebar to select the file containing your test data.;
2. Enter the engine number and its current cycle in the designated fields. 
2. Click on the "Predict" button and wait for the result to appear on the screen. 
""")


from cnn_svr import Cnn_Svr,Cnn_Svr2
st.sidebar.title("User Input")

upload_data= st.sidebar.file_uploader('Upload a file containing the test data')

eng=range(1,101,1)
   
en=st.sidebar.selectbox("Enter the engine number",eng)
cycle=st.sidebar.number_input("Enter the current cycle",min_value=1)


if upload_data is not None:

    columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
    "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
    ,"sensor20","sensor21"]
    test_df = pd.read_csv(upload_data,sep= "\s+", header = None,names=columns)
    st.text('A sample of the uploaded test set')
    st.write(test_df.head(3))
    inputs = {"en":en, "cycle":cycle}
    endpoint = 'http://172.31.29.77:8000/predict' # Specify this path for Dockerization to work
    endpoint2='http://172.31.29.77:8000/m' # Specify this path for Dockerization to work
    if st.button('Predict'):
      
        res=requests.post(url= 'http://172.31.29.77:8000/predict',data=json.dumps(inputs))
        st. subheader(f"The RUL  of engine{en} is= {res.text}")
        mn=requests.post(url='http://172.31.29.77:8000/m',data=json.dumps(inputs))
        
        #####################fig
        import plotly.graph_objects as go 
        fig3 = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = int(res.text),
            mode = "gauge+number+delta",
            title = {'text': "RUL"},
            
            gauge = {'axis': {'range': [int(mn.text),0 ]},
            
             'steps' : [
                 
                 {'range': [20,0,], 'color': "red"}]}))
             
        st.plotly_chart(fig3)


        