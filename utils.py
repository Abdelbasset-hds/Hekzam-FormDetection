import numpy as np
from pdf2image import convert_from_path

def read_pdf(pdf_path) :
    images = convert_from_path(pdf_path)
    return [np.array(img) for img in images]

def save_result():
    pass
