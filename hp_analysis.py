import nltk
import re
from nltk import CFG
import string
import pickle
import os
from tqdm import tqdm, trange

'''
Step 2. Import corpus and segment into sentences
'''

text = ""
f = open('deathly_hallows.txt')
lines = f.readlines()

for line in lines[1:2]:
    text += line[4:].replace('\"', " ").replace('\\', '').strip()


all_sentences = [l.strip().replace("Harry Potter", "HarryPotter") for l in re.split('\.|\?', text) if l]

### NLTK ANALYSIS

# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

'''
Step 2. Tokenize text
'''
tokens = nltk.word_tokenize(text)
tokens = [token.lower() for token in tokens]

list_of_tokens = nltk.pos_tag(tokens)
# print(list_of_tokens)

print("Aantal woorden:" , len(tokens))

'''
Tokenize sentences
'''

sentences = nltk.sent_tokenize(text)
print("Aantal zinnen: ", len(sentences))
# print(sentences)

# normalized_text = "".join([w.lower() for w in text])
# normalized_tokens = nltk.word_tokenize(normalized_text)

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()


# normalized_tokens = [porter.stem(t) for t in tokens]
# normalized_tokens = [lancaster.stem(t) for t in tokens]

# unique_normalized_tokens = set(tokens)
#
# wnl = nltk.WordNetLemmatizer()
# vocabulary = [wnl.lemmatize(t) for t in unique_normalized_tokens]
# print("Aantal vocabulary: ", len(vocabulary))

#
from nltk.parse import stanford
from nltk.parse.stanford import StanfordParser
from nltk.tokenize.stanford import StanfordTokenizer

parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
tokenizer = StanfordTokenizer("/home/spijkervet/stanford/stanford-postagger-full/stanford-postagger-3.9.1.jar")

# Stanford parser uses punctuation!
# processed_sentences = [s.translate(s.maketrans('','', string.punctuation)).lower() for s in sentences]
processed_sentences = [s.lower() for s in sentences]

pickle_name = "hp_trees_parser.pickle"
if not os.path.isfile(pickle_name):
    hp_trees = []
    for s in tqdm(processed_sentences):
        # tree = list(parser.raw_parse(s))
        tree = parser.raw_parse(s)
        # for node in tree:
        #     print(node)
        hp_trees.append(tree)
#
    pickle.dump(hp_trees, open(pickle_name, "wb"))
else:
    hp_trees = pickle.load(open(pickle_name, "rb"))


for parse in hp_trees:
    tree = list(parse)
    for node in tree:
        print(node)
    break


analysis = sentences[0]

# l = st.tag(sentences[0].split())
# print(l)


proc = parser.raw_parse(sentences[0])
for s in proc:
    print(s)
    s.draw()
