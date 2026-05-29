# BioASQ JSON Structure

## questions

### body

- body: the question in text

### documents

- documents : list of links of documents used to search for answer

### ideal_answer

- ideal_answer : text of the correct answer (maybe made by human)

### exact_answer

- exact_answer: short and precise answer instead of a paragaph like in 'ideal_answer'
- value depend on type:
  - type == "list": has a list that has all the answers wrapped in a list (Ex: [['neostigmine'], ['pyridostigmine']])
  - type == "yesno": has as value "yes" or "no" (Ex: 'yes' (Du))
  - type == has a list that has 1 element (Ex: ['Bazex syndrome'])

### concepts

- concepts : links of sources that the answer is based on (ideal_answer is not extractive from the sources)

### type

- type : type of question (categorical)
- possible values :
  - 'list': answer should be a list of elements (Ex: Which drugs were tested in the candor trial?)
  - 'yesno': yes or no question (Ex: Is BCL11B involved in schizophrenia?)
  - 'factoid': answer should be 1 element (Most of them are Which 'questions')(Ex: Which receptor is targeted by telcagepant?)
  - 'summary': answer should be a paragraph (like a definition) (Ex: What is ATAC-seq?)

### id

- id : id of QA (Ex: 55031181e9bde69634000014)

### snippets

#### text

- text : text extracted from document (link in the same snippets)

#### document

- document : document link that the text got from

#### beginSection

- beginSection: from which part the text is extracted from

##### possible values

- 'title': text is extracted from the title of the document
- abstract: text is extracted from the body of the document

#### endSection

- endSection: (same as beginSection since i didnt find a case where beginSection != endSection)

#### offsetInBeginSection

- offsetInBeginSection: start character index of the text (depends of beginSection/endSection)

#### offsetInEndSection

- offsetInEndSection: end character index of the text (depends of beginSection/endSection)

### triples

- triples: list contains triplets of s (subject), p (predicate), and o (object), its a way to represent RDF graphs (s ---(p)---> o, s and o are nodes, p is an edge) (Ex of a triplet: {'p': '[http://www.w3.org/2004/02/skos/core#notation](http://www.w3.org/2004/02/skos/core#notation)',
  's': '[http://linkedlifedata.com/resource/umls/label/A11914653](http://linkedlifedata.com/resource/umls/label/A11914653)',
  'o': 'CDR0000481348'})

#### s (subject)

#### o (object)

#### p (predicate)

---
## Trash columns (to delete)

### duplicate_tmp

- duplicate_tmp: (its a flag for multiple questions that have the same 'id', also i found jsut 1 occurence of 'duplicate_tmp', so we can just delete it with all its subkeys)

### _body

- _body: (same as 'body' but is not answered in 'questions' element, and it also comes with '_type', we can just delete it)

### _type

- _type: (same as '_type', see '_body' above)
