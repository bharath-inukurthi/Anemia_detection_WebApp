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



# Run Locally

Clone the project



```bash
  git clone https://github.com/bharath-inukurthi/Anemia_detection_WebApp.git
```

Go to the project directory

```bash
  cd Anemia_detection_WebApp
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the FastAPI first

```bash
  uvicorn anemia_classification:app
```
Launch the WebApp interface

**NOTE** : run both above and belowe commands in 2 different terminals as both will give you urls to work with

```bash
  python gradioApp.py
```


## Feedback

If you have any feedback, please reach out to us at bharathinukurthi1@gmail.com


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://everything-about-bharath.webflow.io/)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bharath-kumar-inukurthi/)


