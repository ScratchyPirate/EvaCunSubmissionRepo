# this file should take Evacun_out.csv and read the data, processing it with Levenshtein distance, and turning it into 
# clusters.csv, which is an output file where each line holds the lemma, clean value, and then the levenshtein distance 
# between the two

import Levenshtein
import csv
import os

#function defs

#getDists stores each CV preceded by its lemma and followed by the levenshtein dist between the two
def getDists(lemmalist, fill):
    with open(os.getcwd() + '\\Evacun_out_new.csv', newline='', encoding="utf8") as file:
        reader = csv.reader(file)
        lemma_int = 0
        lemma = ""
        for row in reader:
            list_el = row
            if list_el[0] != lemma: #if we're looking at a new lemma
                lemma = lemmalist[lemma_int]
                lemma_int += 1
            else:
                fill.append([lemma, list_el[4], Levenshtein.distance(list_el[4], lemma)]) 
                #adds levenshtein distance between lemma and each CV
    return fill

def main():
    #read file 
    with open(os.getcwd() + '\\Evacun_out_new.csv', newline='', encoding="utf8") as file:
        reader = csv.reader(file)
        #some variables
        lemma = "Lemma"
        comp_list = []
        dists_list = []
        storage_list = []
        lemma_list = []
        frequency = 0
        most_freq = ""
        centroid_list = []

        #for each line in file, add to storage_list. if lemma is same as lemma currently being looked at, add clean value to comp_list
        #else, update lemma, add to comp_list, and continue adding clean values. also grabs centroids by looking at most frequent CVs 
        #within each lemma and puts those in a list
        for row in reader:
            storage_list = row
            if (row[0] != "Lemma"):
                #to pick centroids as we go
                if int(row[5]) > frequency:
                    most_freq = row[4]
                #if we are now looking at the next lemma, reset frequency and most_freq, get next lemma, etc
                if lemma != storage_list[0]:
                    comp_list.append("next")               #append "next" as a flag that is used to write properly to file
                    lemma_list.append(lemma)               #append lemma to lemma_list
                    frequency = 0
                    most_freq = row[4]
                    lemma = storage_list[0]
                    comp_list.append(storage_list[4])      #append CV to comp_list
                    centroid_list.append(most_freq)        #append most_freq to centroid_list
                else:
                    comp_list.append(storage_list[4]) #adding clean value to comp_list so each CV is appended regardless of lemma
        lemma_list.append(lemma)
        centroid_list.append(most_freq)

    #write CVs to output file with the distance to their lemma
    with open(os.getcwd() + "\\Levenshtein Generator\\dist_to_lemmas.csv", 'w', encoding='utf-8', newline='\n') as file:
        fieldnames = ['lemma', ' CV', ' distance',]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        list = getDists(lemma_list, dists_list)
        for i in list:
            writer.writerow({'lemma': i[0], ' CV': i[1], ' distance': i[2]})
            
    #write centroids to output file with distance to each CV
    with open(os.getcwd() + "\\Levenshtein Generator\\clusters_new.csv", 'w', encoding='utf-8', newline='\n') as file:
        fieldnames = ['Lemma', ' CV1', ' CV2', ' distance',]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        lemma_int = 0
        centroid_int = 0
        
        # for j in range(len(comp_list)):
        #     for i in range(len(comp_list) - 1):
        #         writer.writerow({'Lemma': lemma_list[lemma_int], ' CV1': comp_list[j], ' CV2': comp_list[i + 1], ' distance': Levenshtein.distance(comp_list[j], comp_list[i + 1])})
        for i in comp_list:
            j = centroid_list[centroid_int - 1]
            if (i == "next"):
                lemma_int += 1
                centroid_int += 1
            else:
                writer.writerow({'Lemma': lemma_list[lemma_int],' CV1': j, ' CV2': i, ' distance': Levenshtein.distance(i, j)})

if __name__ == "__main__":
    main()
