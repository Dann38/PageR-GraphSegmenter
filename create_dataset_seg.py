import json
from pager import PageModel, PageModelUnit, WordsAndStylesModel, SpGraph4NModel, WordsAndStylesToSpGraph4N, WordsAndStylesToSpDelaunayGraph
from pager.page_model.sub_models.dtype import Style, StyleWord, ImageSegment
import os
import argparse

def get_page_model(type_graph, with_text=True):
    conf = {"with_text": True} if with_text else None
    if type_graph == "4N":
        return  PageModel([
    PageModelUnit("words_and_styles", sub_model=WordsAndStylesModel(), extractors=[], converters={}),
    PageModelUnit("graph", sub_model=SpGraph4NModel(), extractors=[],  converters={"words_and_styles": WordsAndStylesToSpGraph4N(conf) }),
])
    if type_graph == "Delaunay":
        return  PageModel([
    PageModelUnit("words_and_styles", sub_model=WordsAndStylesModel(), extractors=[], converters={}),
    PageModelUnit("graph", sub_model=SpGraph4NModel(), extractors=[],  converters={"words_and_styles": WordsAndStylesToSpDelaunayGraph(conf)}),
])


def is_one_block(word1, word2, blocks):
    for block in blocks:
        if block.is_intersection(word1) and block.is_intersection(word2):
            return 1
    return 0

def get_graph_from_file(file_name, page_model):
    with open(file_name, "r") as f:
        info_img = json.load(f)
    publaynet_rez = info_img["blocks"]
    pager_rez = info_img["additional_info"]
    page_model.from_dict(pager_rez)
    page_model.extract()
    
    graph = page_model.to_dict()

    blocks = [ImageSegment(dict_p_size=bl) for bl in publaynet_rez]
    words = [w.segment for w in page_model.page_units[0].sub_model.words]
    edges_ind = [is_one_block(words[i],words[j], blocks) for i, j in zip(graph["A"][0], graph["A"][1])]
    graph["true_edges"] = edges_ind
    return graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create dataset for PageR from PubLayNet')
    parser.add_argument('--path_dir_jsons', type=str, nargs='?', required=True,
                        help='path json dataset')
    parser.add_argument('--path_rez_json', type=str, nargs='?', required=True,
                        help='path rez json file')
    parser.add_argument('--type_graph', type=str, nargs='?', required=True, help='type graph (4N, Delaunay)')
    
    args = parser.parse_args()
    args.path_dir_jsons, 
    args.path_rez_json
    page_model = get_page_model(type_graph=args.type_graph)
    dataset = {
        "dataset" : []
    }
    files = os.listdir(args.path_dir_jsons)
    N = len(files)
    for i, json_file in enumerate(files):
        try:
            graph = get_graph_from_file(os.path.join(args.path_dir_jsons, json_file), page_model)
            dataset["dataset"].append(graph)
        except:
            print("error in ", json_file)
        print(f"{(i+1)/N*100:.2f} %"+20*" ", end='\r')     
        
    with open(args.path_rez_json, "w") as f:
        json.dump(dataset, f)
        
