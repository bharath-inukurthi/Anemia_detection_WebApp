# Anemia Detection Web App  

## Introduction  
The **Anemia Detection Web App** is a powerful and user-friendly tool designed to identify anemia and estimate hemoglobin levels through conjunctiva images. Leveraging advanced computer vision techniques and machine learning models like YOLO for segmentation and ANN for regression, this application provides quick and accurate results. Its primary goal is to assist users in monitoring their health effortlessly and in real-time.

## Features  
- **Conjunctiva Segmentation:** Utilizes YOLO segmentation to extract conjunctiva regions from user-uploaded images.  
- **Hemoglobin Level Estimation:** Employs an ANN model to predict hemoglobin levels based on segmented image data.  
- **Real-time Processing:** Ensures fast and reliable results for user convenience.  
- **Interactive UI:** Built with Gradio for an intuitive and seamless user experience.  
- **Backend Optimization:** FastAPI ensures efficient handling of image inference requests.


 
# Create a virtual environment

I recommend you to create a virtual environment to avoid compatibilty issues

### macOS/Linux (using Python's venv):
```bash
  python3.11 -m venv anemia_detection
```

```bash
  source anemia_detection/bin/activate
```
### Windows (using Python's venv):

```bash
  python3.11 -m venv anemia_detection
```

```bash
  anemia_detection\Scripts\activate
```


### Using conda:
```bash
  conda create -n anemia_detection python=3.11
```



```bash
  conda activate anemia_detection
```



