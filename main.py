import configuration as cf
from utils import read_pdf, read_json ,save_result
from preprocess.preprocess import Preprocess

def main() :
    images = read_pdf(cf.pdf_path)
    src,rects = read_json(cf.json_path)
    preprocess = Preprocess(images,src)
    rects = preprocess.boxe_correct(rects)
    result = preprocess.draw_boxes(images,rects)
    preprocess.crop_boxes(images,rects)
    save_result(result)

if __name__ == "__main__" :
    main()