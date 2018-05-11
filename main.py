import nltk

text = ""
f = open('deathly_hallows.txt')
lines = f.readlines()

for line in lines[1:2]:
    text += line[4:].replace('"', "").replace('\\', '').strip()

all_sentences = [l for l in text.split('.') if l]
tokens = nltk.word_tokenize(text)
# print(tokens)

# print(all_sentences[0])

hp_grammar = nltk.CFG.fromstring(all_sentences[0])
parser = nltk.ChartParser(hp_grammar)
for tree in parser.parse(sent):
    print(tree)
