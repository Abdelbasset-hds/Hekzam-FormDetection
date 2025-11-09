import numpy as np
from pdf2image import convert_from_path
import json
from pyzbar.pyzbar import decode
import os
import cv2
import configuration as cf

def read_pdf(pdf_path, dpi=300):
    images = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(img) for img in images]

def read_json(json_path):
    """Retourne les coins théoriques des QR codes et toutes les autres boîtes."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        tltl = (data["marker barcode tl page1"]["x"], data["marker barcode tl page1"]["y"])
        tlbr = (data["marker barcode tl page1"]["x"] + data["marker barcode tl page1"]["width"], data["marker barcode tl page1"]["y"] + data["marker barcode tl page1"]["height"])
        trtl = (data["marker barcode tr page1"]["x"],data["marker barcode tr page1"]["y"])
        trbr = (data["marker barcode tr page1"]["x"] + data["marker barcode tr page1"]["width"],data["marker barcode tr page1"]["y"] + data["marker barcode tr page1"]["height"])
        bltl = (data["marker barcode bl page1"]["x"],data["marker barcode bl page1"]["y"])
        blbr = (data["marker barcode bl page1"]["x"] + data["marker barcode bl page1"]["width"], data["marker barcode bl page1"]["y"] + data["marker barcode bl page1"]["height"])
        brtl = (data["marker barcode br page1"]["x"], data["marker barcode br page1"]["y"])
        brbr = (data["marker barcode br page1"]["x"] + data["marker barcode br page1"]["width"],data["marker barcode br page1"]["y"] + data["marker barcode br page1"]["height"])


        rects = []
        for key, value in data.items():
            x = value.get("x", 0)
            y = value.get("y", 0)
            w = value.get("width", 0)
            h = value.get("height", 0)
            rects.append((x, y, w, h))

        return [tltl,tlbr,trtl,trbr,bltl,blbr,brtl,brbr], rects


def read_Qrcode(img):

    qr_code = decode(img)
    src = [None,None,None,None,None,None,None,None]
    for qr in qr_code:
        data = qr.data.decode("utf-8")
        points = [(p.x, p.y) for p in qr.polygon]

        points = sorted(points, key=lambda p: (p[1], p[0]))  
        top = sorted(points[:2], key=lambda p: p[0])  
        bottom = sorted(points[2:], key=lambda p: p[0]) 
        tl, tr = top
        bl, br = bottom

        key_prefix = data.split(',')[0]
        if key_prefix == "hztl":
            src[0] = tl
            src[1] = br
        elif key_prefix == "hztr":
            src[2] = tl
            src[3] = br
        elif key_prefix == "hzbl":
            src[4] = tl
            src[5] = br
        elif key_prefix == "hzbr":
            src[6] = tl
            src[7] = br

    print(f"result : {src}")
    return src

def transform_rects(rects, H):
    
    transformed_rects = []

    for rect in rects:
        x, y, w, h = rect
        corners = np.array([[x, y],
                            [x+w, y],
                            [x+w, y+h],
                            [x, y+h]], dtype=np.float32)

        corners = corners.reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = transformed_corners.reshape(-1, 2)
        xmin, ymin = np.min(transformed_corners, axis=0)
        xmax, ymax = np.max(transformed_corners, axis=0)
        

        transformed_rects.append((int(xmin), int(ymin), int(xmax), int(ymax)))

    return transformed_rects

def mm_to_pixel_list(coords_mm,img):

    img_height_px, img_width_px = img.shape[:2]
    scale_x = img_width_px / cf.img_width
    scale_y = img_height_px / cf.img_height

    coords_px = []
    for c in coords_mm:
        if len(c) == 2:  
            x_px = int(round(c[0] * scale_x))
            y_px = int(round(c[1] * scale_y))
            coords_px.append((x_px, y_px))
        elif len(c) == 4: 
            x_px = int(round(c[0] * scale_x))
            y_px = int(round(c[1] * scale_y))
            w_px = int(round(c[2] * scale_x))
            h_px = int(round(c[3] * scale_y))
            coords_px.append((x_px, y_px, w_px, h_px))
        else:
            raise ValueError("Chaque coordonnée doit être (x,y) ou (x,y,w,h)")
    return coords_px



def src_dst_preprocess(src, dst):
    new_src = []
    new_dst = []

    for s, d in zip(src, dst):
        if d is not None:
            new_src.append(s)
            new_dst.append(d)

    return new_src, new_dst




def save_result(pages, base_folder="output", sub_folder="preprocess"):
    output_folder = os.path.join(base_folder, sub_folder)
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(pages):
        filename = os.path.join(output_folder, f"{i}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"{len(pages)} images enregistrées dans {output_folder}")
