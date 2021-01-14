import streamlit as st
import cv2
from PIL import Image
import numpy as np



body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

def detect_body(img):
    
	img = np.array(img)
	body_img = img.copy()
	gray = cv2.cvtColor(body_img, cv2.COLOR_BGR2GRAY)
	body_rects = body_cascade.detectMultiScale(gray, 1.1, 1) 
    
	for (x,y,w,h) in body_rects: 
		cv2.rectangle(body_img, (x,y), (x+w,y+h), (135, 50, 168), 10) 
        
	return body_img
	


st.title("Pedistrian detection")

html_temp = """
<body style="background-color:red;">
<div style="background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;">Pedistrian Recognition WebApp</h2>
</div>
</body>
"""
st.markdown(html_temp, unsafe_allow_html=True)

image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
if image_file is not None:
	img = Image.open(image_file)
	st.text("Original Image")
	st.image(img)

if st.button("Compute"):
	result_img= detect_body(img)
	st.image(result_img)
	