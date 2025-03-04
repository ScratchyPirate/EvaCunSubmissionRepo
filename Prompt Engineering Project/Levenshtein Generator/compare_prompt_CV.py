import Levenshtein
import csv

promptedCV = "hi" # not entirely sure how to get the prompted CVs ***********************************************


#comp_prompt takes each prompted CV and returns the lemma of the "closest" CV in the data 
# we've already looked at
def comp_prompt(promptedCV):
    #take a prompted CV and find distance between all other CVs to find which lemma it should belong to
    with open('Levenshtein\clusters.csv', newline='') as file:
        reader = csv.reader(file)
        min_dist = 100
        found_lemma = ""
        for row in reader: #finds smallest distance between prompted CV and stored ones
            if min_dist > Levenshtein.distance(promptedCV, row[2]):
                min_dist = Levenshtein.distance(promptedCV, row[2])
                found_lemma = row[0]
    return found_lemma