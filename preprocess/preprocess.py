import cv2
import numpy as np
from utils import read_Qrcode ,src_dst_preprocess, mm_to_pixel_list,transform_rects
import configuration as cf
import os

class Preprocess:
    def __init__(self, pages, src):
        self.pages = pages
        self.src = src
    
    def boxe_correct(self,rects):

        result = []
        i = 0
        for page in self.pages:
            
            dst = read_Qrcode(page)  
            src = mm_to_pixel_list(self.src,page)
            src , dst = src_dst_preprocess(src,dst)

            H, _ = cv2.findHomography(np.array(src, dtype=np.float32),np.array(dst, dtype=np.float32))
            rects_px = mm_to_pixel_list(rects,page)
            rects_px = transform_rects(rects_px,H)


            print(f"source : {src}, destination : {dst}")
            print(f"matrice : {H} ")
            print(f"Page size: {page.shape}")

            result.append(rects_px)
            print("**************************")
            print(f"rect {i} : {rects_px}")
            i += 1
            print("**************************")
        
        return result
    
    def draw_boxes(self,pages,coordinates): 
        result = list() 
        for cordinate,page in zip(coordinates,pages) : 
            img = page.copy() 
            for (xmin, ymin, xmax, ymax) in cordinate:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            result.append(img) 
        return result
        


    def crop_boxes(self, pages, coordinates, output_dir="output/crop", target_size=(28, 28), margin=3):

        os.makedirs(output_dir, exist_ok=True)

        for page_idx, (page, boxes) in enumerate(zip(pages, coordinates)):
            page_dir = os.path.join(output_dir, str(page_idx))
            os.makedirs(page_dir, exist_ok=True)

            for box_idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                h, w = page.shape[:2]

                xmin = max(0, xmin + margin)
                ymin = max(0, ymin + margin)
                xmax = min(w, xmax - margin)
                ymax = min(h, ymax - margin)

                if xmax <= xmin or ymax <= ymin:
                    continue

                crop = page[ymin:ymax, xmin:xmax]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                if np.mean(gray) > 127:
                    gray = 255 - gray
                    
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

                canvas = np.zeros(target_size, dtype=np.uint8)

                h, w = gray.shape
                scale = min(target_size[0] / h, target_size[1] / w)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

                x_offset = (target_size[1] - new_w) // 2
                y_offset = (target_size[0] - new_h) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

                crop_path = os.path.join(page_dir, f"box_{box_idx}.png")
                cv2.imwrite(crop_path, canvas)