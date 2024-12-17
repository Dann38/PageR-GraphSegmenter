import os
import argparse
import numpy as np
from pager import (PageModel, PageModelUnit,
                   ImageModel, ImageToWordsAndStyles,
                   WordsAndStylesModel, PhisicalModel, 
                   WordsAndStylesToGNNBlocks)
from pager.page_model.sub_models.dtype import ImageSegment
from pager.metrics.uoi import segmenter_UoI as UoI, AP_and_AR_from_TP_FP_FN as AP_and_AR, TP_FP_FN_UoI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test dataset')
    parser.add_argument('--path_test_dataset', type=str, nargs='?', required=True)
    # parser.add_argument('--name_model', type=str, nargs='?', required=True)
    args = parser.parse_args()

    phis_model = PageModel([
        PageModelUnit(id="phis", 
                      sub_model=PhisicalModel(), 
                      extractors=[], 
                      converters={})])
    image_model = PageModel(page_units=[
        PageModelUnit(id="image_model", 
                      sub_model=ImageModel(), 
                      extractors=[], 
                      converters={}),
        PageModelUnit(id="words_and_styles_model", 
                      sub_model=WordsAndStylesModel(), 
                      extractors=[], 
                      converters={"image_model": ImageToWordsAndStyles(conf={"lang": "eng+rus", "psm": 4, "oem": 3, "k": 4})}),
        PageModelUnit(id="phisical_model", 
                      sub_model=PhisicalModel(), 
                      extractors=[], 
                      converters={"words_and_styles_model": WordsAndStylesToGNNBlocks()})
        ]) 

    path = args.path_test_dataset
    files = [f for f in os.listdir(path) if f[-3:]=='jpg']
    UoI_array = []
    TP_50, FN_50, FP_50 = 0,0,0
    TP_95, FN_95, FP_95 = 0,0,0
    for i,file in enumerate(files):
        
        image_path = os.path.join(path, file)
        json_path = os.path.join(path, file+'.json')
        phis_model.read_from_file(json_path)
        image_model.read_from_file(image_path)
        image_model.extract()
        print(i)
        pred_seg = [block.segment for block in phis_model.page_units[-1].sub_model.blocks]
        true_seg = [block.segment for block in image_model.page_units[-1].sub_model.blocks]
        uoi = UoI(pred_seg, true_seg)
        UoI_array.append(uoi) 
        TP_, FP_, FN_ = TP_FP_FN_UoI(pred_seg, true_seg)  
        TP_50+=TP_
        FN_50+=FN_
        FP_50+=FP_
        TP_2, FP_2, FN_2 = TP_FP_FN_UoI(pred_seg, true_seg, alpha=0.95)
        TP_95+=TP_2
        FN_95+=FN_2
        FP_95+=FP_2
        print(f"UoI:{uoi}\t TP[0.5]:{TP_}\t FN[0.5]:{FN_}\t FP[0.5]:{FP_}\t TP[0.95]:{TP_2}\t FN[0.95]:{FN_2}\t FP[0.95]:{FP_2}")

    print(f"UoI:{np.mean(UoI_array):.3f}")
    AP_50, AR_50 = AP_and_AR(TP_50, FP_50, FN_50) 
    AP_95, AR_95 = AP_and_AR(TP_95, FP_95, FN_95)
    print(f"AP UoI[0.5]:{AP_50:.3f}")
    print(f"AR UoI[0.5]:{AR_50:.3f}")
    print(f"AP UoI[0.95]:{AP_95:.3f}")
    print(f"AR UoI[0.95]:{AR_95:.3f}")
