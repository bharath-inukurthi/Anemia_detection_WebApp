from fastapi import FastAPI, UploadFile,File
from ultralytics import YOLO
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
import pickle as p
import tensorflow.keras.models as models
ANN=models.load_model("mae_1.6(0.11)_similar")
with open("scaler.pkl",'rb') as f2:
    scaler=p.load(f2)
with open("tar_scaler.pkl",'rb') as f1:
    tar_scaler=p.load(f1)
app=FastAPI()
model=YOLO("eye_seg_model.pt")
cache='image.jpeg'
@app.post("/predict/")
async def predict(file:UploadFile=File(...)):
    image=await file.read()
    #cv2.imwrite(cache,np.array(Image.open(BytesIO(image))))
    im=np.array(Image.open(BytesIO(image)))
    im=cv2.resize(im,(480,640))
    img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    cv2.imwrite(cache, img)
    results=model.predict(cache)
    try:
        mask=results[0].masks.data[0].cpu().numpy().astype("uint8") * 255
        seg=cv2.bitwise_and(img,img,mask=mask)
        B,G,R=cv2.split(seg)
        B,G,R=np.sum(B),np.sum(G),np.sum(R)
        T=B+G+R
        b_percent=(B/T)*100
        g_percent=(G/T)*100
        r_percent=(R/T)*100
        inputs=scaler.transform(np.array([r_percent,g_percent,b_percent]).reshape(1,-1))
        prediction=ANN.predict(inputs)
        hgl=tar_scaler.inverse_transform(prediction)[0][0]-1

        status="Anemiac" if hgl-1<11 else "Non Anemiac"
        return {"hgl":f"{hgl-1:.2f}g/dl",
                "status":status}
    except AttributeError as e:
        return{"hgl":"Oops!",
               "status":"Unable to capture conjuctiva recapture the image"}

