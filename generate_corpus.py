from tqdm.auto import tqdm
from datasets import load_dataset
import nltk
from nltk import tokenize

from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/input/osilab-nlp")
    parser.add_argument("--corpus_dir", type=str, default="/input/osilab-nlp/wikipedia")
    args = parser.parse_args()
    
    nltk.download('punkt')
    os.makedirs(args.corpus_dir, exist_ok=True)
    
    wiki = load_dataset('wikipedia', '20200501.en', split='train', cache_dir=args.cache_dir)
    with open(os.path.join(args.corpus_dir, "corpus.txt"), "w", encoding="utf-8") as fp:
        for idx, document in tqdm(enumerate(wiki)):
            document = document["text"].replace("\n", " ")
            document = tokenize.sent_tokenize(document)
            for sentence in document:
                fp.write(sentence)
                fp.write("\n")
            fp.write("\n")
            
#             if idx == 1000:
#                 break
            
if __name__ == "__main__":
    main()