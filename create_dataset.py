from pager import PageModel, PageModelUnit
from pager.page_model.sub_models import ImageModel, WordsAndStylesModel, SpGraph4NModel
from pager.page_model.sub_models import ImageToWordsAndCNNStyles,  WordsAndStylesToSpGraph4N
from publaynet_reader import PubLayNetDataset 
import argparse
import os


def get_words_and_styles(img_path):
    page.read_from_file(img_path)
    page.extract()
    return page.page_units[-1].sub_model.to_dict(is_vec=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create dataset for PageR from PubLayNet')
    parser.add_argument('--path_publaynet', type=str, nargs='?', required=True,
                        help='path publaynet')
    parser.add_argument('--path_words_and_styles', type=str, nargs='?', required=True,
                        help='path blocks, words and styles dataset')
    parser.add_argument('--start', type=int, nargs='?', required=True,
                        help='category exist') 
    parser.add_argument('--finish', type=int, nargs='?', required=True,
                        help='category will exist')               
    args = parser.parse_args()
    PATH_STYLE_MODEL = os.environ["PATH_STYLE_MODEL"]
    page = PageModel([
    PageModelUnit(id="image", sub_model=ImageModel(), converters={}, extractors=[]),
    PageModelUnit(id="word_and_style", sub_model=WordsAndStylesModel(), converters={"image": ImageToWordsAndCNNStyles(conf={"path_model": PATH_STYLE_MODEL,"lang": "eng+rus", "psm": 4, "oem": 3, "k": 4 })}, extractors=[]),
])
    pln_ds = PubLayNetDataset(args.path_publaynet, args.path_words_and_styles)
    pln_ds.create_tmp_annotation_jsons(path_tmp_dataset=args.path_words_and_styles, 
                                       fun_additional_info=get_words_and_styles, 
                                       start_min_category= args.start, finish_min_category=args.finish)
    
    