import argparse
from data_utils import reverse_dict, export_dict, write_file
import json
import os
import pandas as pd

def main():

    parser = argparse.ArgumentParser()     
    parser.add_argument("INPUT_DIR", type=str)
    parser.add_argument("OUTPUT_DIR", type=str)
    args = parser.parse_args()

    """
        Probably train_PlayMusic_full.json gives an unicodeError:
            resave rewritting the quotation marks of the lines 8896-8897    
    """

    intents = [
        "AddToPlaylist",
        "GetWeather",  
        "RateBook",   
        "SearchCreativeWork",
        "BookRestaurant",
        "PlayMusic",
        "SearchScreeningEvent",
    ]

    files = [
        ("train_{}_full.json", "train_full"),
        ("validate_{}.json", "validate"),
    ]

    for (in_file, out_file) in files:
        list_queries = []
        list_intents = []
        
        for intent in intents:    
            file_ = in_file.format(intent)
            current_file = os.path.join(args.INPUT_DIR, file_)
            
            with open(current_file, "r", encoding="utf-8") as f:
                query_dataset = json.load(f)
            
            for query in query_dataset[intent]:
                list_segments = []
                
                for segment in query["data"]:
                    list_segments.append(segment["text"].rstrip())
                
                new_segments = "".join(list_segments).rstrip()
                if "\n" in new_segments:
                    new_segments = new_segments.replace("\n", " ")
                list_queries.append(new_segments)
                list_intents.append(intent)
        
        unique_label = sorted(list(set(list_intents)))
        label_id = {unique_label[id_] : id_ for id_ in range(len(unique_label))}
        list_intents = list(map(lambda key: label_id[key], list_intents))
        id_label = reverse_dict(label_id)

        write_file(os.path.join(args.OUTPUT_DIR, f"{out_file}"), list_queries, list_intents, "query", "intent")
        export_dict(os.path.join(args.OUTPUT_DIR, "labels"), id_label)

    
if __name__ == "__main__":
    main()