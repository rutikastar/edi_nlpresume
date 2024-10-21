import spacy
import pickle
import pymupdf as fitz

#reading the pickle data 
#train_data1 used as test_data_1 for testing the nlp_model_2
train_data1 = pickle.load(open('train_data.pkl','rb'))
print(train_data1[0])

#second model loaded into nlp_model2 (trained before in jupyternbk and stored)
nlp_model2 = spacy.load('nlp_ner_model2')

#final nlp model (model 2 - better model) tested with train_data_1, which becomes the test data for it now
print("\n\n")
print("Testing for NLP model: ")
test_data2 = train_data1
doc = nlp_model2(test_data2[0][0])
for ent in doc.ents:
    print(f"{ent.label_.upper():{30}}-{ent.text}")


fname = 'Alice Clark CV.pdf'
doc = fitz.open(fname)
text = ""
for page in doc:
    text += page.get_text()
pdftxt = " ".join(text.split('\n'))

print("\n\n")
print("This is the alica clark dummy resume")
print(pdftxt)

# Applying the 2nd model on the pdf text extracted
doc = nlp_model2(pdftxt)
for ent in doc.ents:
    print(f'{ent.label_.upper():{30}}- {ent.text}')

