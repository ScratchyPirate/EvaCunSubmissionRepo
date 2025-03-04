#
#   File: Data.py
#   Description: Contain System classes and methods revolving around
#       retrieval and manipulation of datasets related to the project.
#       Provides the information in such datasets to the rest of the 
#       project.
#

# Imports
#   Local
from DataHelpers import CleanValue, Lemma, Prompt, PromptType, Language, WordType, CVExampleSentenceMaskToken
from Messages import SystemMessages

#   External
from enum import Enum
import os
from pandas import DataFrame
import pandas as pd
import sys
from typing import Final
import warnings
import numpy as np

#
#   Class focused on retrieval of, storage of, and access to lemmatization data
#
class LemmaDictionary:

    # 
    #   Fields
    #
    # For general use:
    __lemmaDictionary: dict # Represents the dictionary of lemmas and their corresponding CVs available. The key value pairs are in format (lemma name, Lemma object)

    # For I/O
    __loaded: bool   # Represents whether the dictionary has been loaded with data from a .csv file
    __storePath: str # Represents the last path to a .csv file where the dictionary is stored from the last time
                     # the store method ran.
    
    # For querying
    __lemmaDataFrame: DataFrame # Represents the dataframe of lemmas, a representation of the .csv file. Used for querying.

    # For testing
    __totalTestCount: int    # Represents the total tests done across all lemmas in the dictionary
    __totalCorrectTestCount: int # Represents the total number of tests that were correct that were done across all lemmas.
    

    #
    #   Constructors
    #
    # Default constructor
    def __init__(self):
        self.__lemmaDictionary = dict()

        # Set the maximum recursion limit to 30000 (necessary for quicksort)
        sys.setrecursionlimit(30000)

        # Init load and store vars
        self.__loaded = False
        self.__storePath = ""

        # Init dataframe to none, representing it being uninitialized.
        self.__lemmaDataFrame = None

        # Test variables
        self.__totalTestCount = 0
        self.__totalCorrectTestCount = 0
        
    #
    #   Setters & Getters
    #
    def GetLemmaDictionary(self) -> dict:
        return self.__lemmaDictionary.copy()
    
    #
    #   Methods
    #

    #
    #       Related to Lemma Dictionary Methods
    #

    """
    #   Given an inputted key, attempts to find a Lemma with that key
    #       in the dictionary.
    #       Returns the Lemma if found
    #       Returns None otherwise
    """
    def GetLemma(self, key: str) -> Lemma:
         if (key in self.__lemmaDictionary.keys()):
            return self.__lemmaDictionary[key]
         else:
            SystemMessages.LD_LNotFound(key)
            return None
         
    """ 
    #   Given a string representing a Lemma, if that Lemma doesn't exist within the dictionary's
    #       Lemma dictionary, then a new Lemma object is created with that string and inserted
    #       into the dictionary. Otherwise, the number of occurences of an existing Lemma
    #       are incremented instead.
    """    
    def InsertLemma(self, key: str):
        if (key in self.__lemmaDictionary.keys()):
            self.__lemmaDictionary[key].IncrementOccurences()
        else:
            newLemma: Lemma = Lemma(key)
            
            # Subscribe to lemma events
            newLemma.SubscribeTestCorrect(self.__IncrementTotalCorrectTestCount)
            newLemma.SubscribeTestCorrect(self.__IncrementTotalTestCount)
            newLemma.SubscribeTestIncorrect(self.__IncrementTotalTestCount)

            self.__lemmaDictionary[key] = newLemma

    """
    #   Updates all relative frequencies of lemmas and their CVs in the lemma dictionary
    """
    def UpdateFrequencies(self):

        # Locals
        totalAppearance: int = 0

        # Get the total number of lemma appearances
        for l in self.__lemmaDictionary.values():
            totalAppearance += l.GetOccurences()

        # For each lemma, update its frequencies and each of its CV's frequencies
        for l in self.__lemmaDictionary.values():
            l.SetRelativeFrequency(l.GetOccurences()/totalAppearance)
            l.UpdateCVFrequencies()

    # Total tests on all lemmas
    def TotalTestCount(self) -> int:
        return self.__totalTestCount
    # Total correct tests on all lemmas
    def TotalCorrectTestCount(self) -> int:
        return self.__totalCorrectTestCount
    # Total incorrect tests on all lemmas
    def TotalIncorrectTestCount(self) -> int:
        return self.__totalTestCount - self.__totalCorrectTestCount
    # Total accuracy across tests on all lemmas
    def TotalTestAccuracy(self) -> float:
        if self.__totalTestCount == 0: 
            return 0
        return self.__totalCorrectTestCount / self.__totalTestCount

    """
    #       Related to Dataset Retrieval and Storage Methods
    #
    #   Loads lemmas into the lemma dictionary based on the inputted file path if the file exists.
    #       Assumes the format of lemmatization_train_no_ids.csv, the file used in training
    #   Returns:
    #       True if successful in loading the lemmas, their properties, their CVs, and their properties
    #       False otherwise
    """
    class __LemmaDatasetColumnHeaderNames(Enum):
        FRAGMENT_ID = 0,
        FRAGMENT_LINE_NUM = 1,
        INDEX_IN_LINE = 2,
        WORD_LANGUAGE = 3,
        DOMAIN = 4,
        VALUE = 5,
        CLEAN_VALUE = 6,
        LEMMA = 7
    def LoadDictionary(self, filePath: str, debug: bool) -> bool:
        
        # If the dictionary is already loaded, return false
        if self.__loaded:
            return False

        # Locals
        streamReader = None
        buffer: str = ""
        bufferTokens: list = []
        tempInt: int = 0
        endOfFile = False
        columnHeads: dict = {
            LemmaDictionary.__LemmaDatasetColumnHeaderNames.FRAGMENT_ID : -1,
            LemmaDictionary.__LemmaDatasetColumnHeaderNames.FRAGMENT_LINE_NUM : -1,
            LemmaDictionary.__LemmaDatasetColumnHeaderNames.INDEX_IN_LINE : -1,
            LemmaDictionary.__LemmaDatasetColumnHeaderNames.WORD_LANGUAGE : -1,
            LemmaDictionary.__LemmaDatasetColumnHeaderNames.DOMAIN : -1,
            LemmaDictionary.__LemmaDatasetColumnHeaderNames.VALUE : -1,
            LemmaDictionary.__LemmaDatasetColumnHeaderNames.CLEAN_VALUE : -1,
            LemmaDictionary.__LemmaDatasetColumnHeaderNames.LEMMA : -1,
        }
        exampleSentenceTokens: list = []    # storage for current sentence represented by currentfragmentlinenum
        exampleSentence: str = ""   # storage for current sentence once fully compiled
        lcvInSentence: list = [] # storage for lemma cv pairs in example sentence

        oldFragmentLineNum: str = "-1"  # storage for last line num index
        currentFragmentLineNum: str = "-1"  # storage for current line num index
        currentIndexInLine: str = "-1" # storage for index in line
        lemmaLanguage: Language = "" # represents the language in a data row
        cleanValueToken: str = "" # represents the raw cv retrieved from a data row
        lemmaValueToken: str = ""# reprsents the raw lemma retrieved from a data row
        valueToken: str = ""    # represents the raw value retrieved from a data row

        # Attempt to open the file from the path provided.
        #   Return false if file cannot be opened/isn't found
        if (not os.path.exists(filePath)):
            SystemMessages.FileNotFound(filePath)
            return False
        else:
            streamReader = open(filePath, "r", encoding="utf8")

        # Read the first line in order to understand which column represents what type of tokens
        buffer = streamReader.readline()
        if (debug):
            print(buffer)

        #      If end of file, return (file is empty).
        if len(buffer) == 0:
            SystemMessages.LD_InvalidFileFormat(filePath)
            streamReader.close()
            return False
        bufferTokens = buffer.split(",")
        if (debug):
            print(bufferTokens)
        for i in range(len(bufferTokens)):
            if "fragment_id" in bufferTokens[i]:
                columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.FRAGMENT_ID] = i
            elif "fragment_line_num" in bufferTokens[i]:
                columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.FRAGMENT_LINE_NUM] = i
            elif "index_in_line" in bufferTokens[i]:
                columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.INDEX_IN_LINE] = i
            elif "word_language"in bufferTokens[i]:
                columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.WORD_LANGUAGE] = i
            elif "domain" in bufferTokens[i]:
                columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.DOMAIN] = i
            elif "clean_value" in bufferTokens[i]:
                columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.CLEAN_VALUE] = i
            elif "value" in bufferTokens[i]:
                columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.VALUE] = i
            elif "lemma" in bufferTokens[i]:
                columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.LEMMA] = i
                    
        # If any of the columns are missing, then return and make a system message that the dataset couldn't be loaded
        for index in columnHeads.values():
            if index == -1:
                SystemMessages.LD_InvalidFileFormat(filePath)
                streamReader.close()
                return False
        
        # Begin reading line by line, perform actions based on what the line encountered means
        while (not endOfFile):

            # Read the next line.
            buffer = streamReader.readline()      
            if (debug):
                print(buffer)

            # If buffer is empty, endOfFile = true, break
            if len(buffer) == 0:
                endOfFile = True
            else:

                # Parse buffer
                bufferTokens = buffer.split(",")
                # Case based on index in buffer tokens
                for i in range(len(bufferTokens)):
                    
                    # fragment id case
                    if columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.FRAGMENT_ID] == i:
                        continue

                    # fragment line number case 
                    elif columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.FRAGMENT_LINE_NUM] == i:

                        # Set sentence line num
                        oldFragmentLineNum = currentFragmentLineNum
                        currentFragmentLineNum = bufferTokens[i]

                    # index in line case
                    elif columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.INDEX_IN_LINE] == i:
                        
                        # Get current index in line
                        currentIndexInLine = bufferTokens[i]

                    # language case
                    elif columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.WORD_LANGUAGE] == i:

                        match bufferTokens[i]:
                            case "AKKADIAN":
                                lemmaLanguage = Language.AKKADIAN
                            case "EMESAL":
                                lemmaLanguage = Language.EMESAL
                            case "SUMERIAN":
                                lemmaLanguage = Language.SUMERIAN
                            case _:
                                lemmaLanguage = Language.UNKNOWN

                    # value case
                    elif columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.VALUE] == i:
                        valueToken = bufferTokens[i]
                        if ("-" in valueToken):
                            valueToken.replace("-","")

                    # clean value case
                    elif columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.CLEAN_VALUE] == i:

                        # Scrub dashes from values
                        cleanValueToken = bufferTokens[i] 
                        if ("-" in cleanValueToken):
                            cleanValueToken = "".join(cleanValueToken.split("-"))
                            cleanValueToken.replace("-","")
                    
                    # lemma value case
                    elif columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.LEMMA] == i:
                        lemmaValueToken = bufferTokens[i]
                        lemmaValueToken.replace("-","")
                    
                    # domain case
                    elif columnHeads[LemmaDictionary.__LemmaDatasetColumnHeaderNames.DOMAIN] == i:
                        continue

                # end buffer parsing

                # Process Buffer
                # If any value isn't initialized, break
                if (len(lemmaValueToken) == 0 or (len(cleanValueToken) == 0 and len(valueToken) == 0)) or currentFragmentLineNum == "-1":
                    if (debug):
                        print("--Invalid buffer encountered")
                        print(bufferTokens)
                else:
                    #   Scrub from lemma "I" or "II" character (I and II don't appear in the actual akkadian dictionary)
                    # if " I" in lemmaValueToken:
                    #     if "II" in lemmaValueToken:
                    #         tempInt = len(lemmaValueToken) - 3
                            
                    #     else:
                    #         tempInt = len(lemmaValueToken) - 2
                    #     lemmaValueToken = lemmaValueToken[0:tempInt]

                    #   Insert lemma
                    self.InsertLemma(lemmaValueToken)
                    #   Adjust lemma language
                    self.__lemmaDictionary[lemmaValueToken].SetLanguage(lemmaLanguage)
                    #   Insert CV
                    #   If CV token wasn't found, use value instead
                    if (len(cleanValueToken)==0):
                        cleanValueToken = valueToken
                    self.__lemmaDictionary[lemmaValueToken].InsertCV(cleanValueToken)

                    # If there was a previous sentence started and the fragment line num isn't equal to the 
                    #   one found in the buffer AND there is more than one token in the example sentence tokens,
                    #   compile the old example sentence. Store in lemmas CVs found in
                    #   example sentence.
                    #   
                    if (oldFragmentLineNum != -1 and currentFragmentLineNum != -1) and oldFragmentLineNum != currentFragmentLineNum:

                        if (len(exampleSentenceTokens)>1):
                            exampleSentence = " ".join(exampleSentenceTokens)
                            for lcv in lcvInSentence:
                                self.__lemmaDictionary[lcv[0]].InsertCVExampleSentence(lcv[1], exampleSentence)
                        exampleSentence = ""
                        exampleSentenceTokens = []
                        lcvInSentence = []

                    #   Add the lemma cv pair to example sentence
                    lcvInSentence.append((lemmaValueToken, cleanValueToken))

                    #   Append to the end of the example sentence the CV found
                    #       If the IndexInLine of the row is greater than the number of sentence tokens,
                    #       i.e. there are missing words in the sentence between the previous row and the
                    #       current row, add mask tokens inbetween
                    while (int(currentIndexInLine) - 1 > len(exampleSentenceTokens)):
                        exampleSentenceTokens.append(CVExampleSentenceMaskToken)
                    exampleSentenceTokens.append(cleanValueToken)

                    # Reset locals used
                    cleanValueToken = ""
                    lemmaValueToken = ""
                    
        # Update 
        self.UpdateFrequencies()

        # Close the reader and return true when finished
        streamReader.close()

        # Set loaded to true
        self.__loaded = True
        return True

    """
    #   Loads levenshtein distance from each cv to its lemma parent and stores it in an
    #       already loaded lemma dictionary
    #   Returns:
    #       True if successful
    #       False otherwise
    """
    def LoadLevenshteinDistanceToLemma(self, filePath: str, debug: bool) -> bool:

        # If the dictionary isn't already loaded, return false
        if not self.__loaded:
            return False

        # Locals
        streamReader = None
        readLine: str
        tokens: list = []
        tempLemmaTag: str
        tempCVTag: str
        tempDistanceTag: int
        endOfFile: bool = False

        # Attempt to open the file from the path provided.
        #   Return false if file cannot be opened/isn't found
        if (not os.path.exists(filePath)):
            SystemMessages.FileNotFound(filePath)
            return False
        else:
            streamReader = open(filePath, "r", encoding="utf8")

        #   Dump first line. If empty, return
        readLine = streamReader.readline()
        if (len(readLine) == 0):
            SystemMessages.LD_InvalidFileFormat(filePath)
            streamReader.close()
            return False

        # Until there are no more lines, read in data and assign distance from cv to lemma found in lemma dictionary if able
        # Format: lemma: str, cv: str, distance: int
        while (not endOfFile):

            # Read the next line
            readLine = streamReader.readline()

            # If the line is empty, we've reached the end of the file. Return
            if len(readLine) == 0:
                endOfFile = True

            # Otherwise, parse the line and assign levenshtein distance
            tokens = readLine.split(',')

            # If there are 3 tokens, continue
            if len(tokens) >= 3:

                # If the lemma exists in the dictionary
                if self.__lemmaDictionary.__contains__(tokens[0]):

                    # If the cv is within the target lemma
                    if self.__lemmaDictionary[tokens[0]].GetCV(tokens[1]) != None:

                        # Set the distance of the CV to the parent
                        self.__lemmaDictionary[tokens[0]].GetCV(tokens[1]).SetLevenshteinDistanceToParent(str(tokens[2]))

        # Close the reader and return true when finished
        streamReader.close()
        return True
    
    """
    #   Loads levenshtein distance from each cv to each of its sibling cvs and stores it in the 
    dictionary of cvs and distances within each lemma
    #   Returns:
    #       True if successful
    #       False otherwise
    """
    def LoadLevenshteinDistanceToSiblings(self, filePath: str, debug: bool) -> bool:

        # If the dictionary isn't already loaded, return false
        if not self.__loaded:
            return False
        
        # Locals
        streamReader = None
        readLine: str
        tokens: list = []
        lemma: str
        cv: CleanValue
        endOfFile: bool = False
        distance: float
        dist_list: list = []

        # Attempt to open the file from the path provided.
        #   Return false if file cannot be opened/isn't found
        if (not os.path.exists(filePath)):
            SystemMessages.FileNotFound(filePath)
            return False
        else:
            streamReader = open(filePath, "r", encoding="utf8")
        
        #   Dump first line. If empty, return False
        readLine = streamReader.readline()
        if (len(readLine) == 0):
            SystemMessages.LD_InvalidFileFormat(filePath)
            streamReader.close()
            return False
        
        # Until there are no more lines, read in data and assign distance from cv to lemma found in lemma dictionary if able
        # Format: 
        while (not endOfFile):

            # Read the next line
            readLine = streamReader.readline()

            if (readLine == ""):
                endOfFile = True
            else:
                # Otherwise, parse the line and assign levenshtein distance
                tokens = readLine.split(',')
                distance = float(tokens[3])

                # If there are 4 tokens, continue (lemma, cv1, cv2, distance)
                if len(tokens) >= 4:

                    # If the lemma exists in the dictionary
                    if self.__lemmaDictionary.__contains__(tokens[0]):
                        lemma = self.__lemmaDictionary.get(tokens[0])

                        # If the cv is within the target lemma
                        if lemma.GetCV(tokens[1]) != None:
                            cv = lemma.GetCV(tokens[1])

                            # Go into lemma to get cv, then dictionary and set the distance
                            cv.SetLevenshteinDistanceToSibling(str(tokens[2]), distance)
                            dist_list.append(distance)

        # Close the reader and return true when finished
        streamReader.close()
        return True

    """
    #   Stores the lemma dictionary and its contents onto the inputted file path.
    #   Creates/overrides file and writes to path as if it were a .csv file
    #   
    #   *Assumes .csv isn't in the file's name inputted. (appends filetype tag to end of filename)
    #
    #   Output file datacolumns and their meanings
    #   Description of each column:
    #         Lemma: The name of the lemma associated with the data in the row
    #         L_appearances: The number of appearances the lemma has in the dataset
    #         L_language: The language of the lemma
    #         L_word_type: The word type of the lemma (noun, verb, etc.)
    #         Clean Value: The name of the clean value associated with the lemma in the row
    #         CV_appearances: The number of appearances the clean value has in the dataset when associated with the row's lemma
    #         T_L_appearances: The total number of all lemma appearances in the dataset (constant for all rows).
    #         P_L_appearances: The percent appearance of the row's Lemma in relation to the total number of lemmas (L_appearances/T_L_appearances)
    #         P_CV_appearances: The percent appearance of the row's Clean Value in relation to the total number of clean value appearances associated with the lemma (CV_appearances/L_appearances)
    #         Example Sentences: Examples of CV being used in a sentence. Format: [sentence1: str; sentence2: str; ...]
    #         Levenshtein Distance to Siblings: Levenshtein distance to siblings as token pairs in a list. Format: [(siblingCV1: str| distance: int); (siblingCV2: str| distance: int); ...]
    #         Levenshtein Distance to Paret: Levenshtein distance to parent.    
    #     Column Names:   |Lemma              |L_appearances       |L_language          |L_word_type            |Clean Value         |CV_appearances           |   T_L_appearances |   P_L_appearances |   P_CV_appearances| Example_Sentences  | Levenshtein Distance to Siblings  |   Levenshtein Distance to Parent }
    #     Column Datatype:| (string)          | (int)              | (string)           | (string)              | (string)           |    (int)                |   (int)           |   (float)         |   (float)         | (list<str>)        | list[(str; int)]                  |   (int)
    #       |
    #   Returns
    #       True if successful
    #       False otherwise
    """
    __outFileColumnHeaders = [
        'Lemma',
        'L_appearances',
        'L_language',
        'L_word_type',
        'Clean Value',
        'CV_appearances',
        'T_L_appearances',
        'P_L_appearances',
        'P_CV_appearances',
        'Example Sentences',
        'Levenshtein Distance to Siblings',
        'Levenshtein Distance to Parent'
    ]
    def StoreDictionary(self, targetDirectory: str, fileName: str, debug: bool) -> bool:

        # Locals
        writer = None
        lDictCopy: dict = dict()
        totalLAppearances: int = 0
        tempDict: dict = dict()
        temp_l: str = ""
        temp_cv: str = ""
        tempLemmaApearances: int = 0
        tempCVAppearances: int = 0
        tempPLemmaAppearances: float = 0
        tempPCVAppearances: float = 0
        tempLLanguage: Language = Language.UNKNOWN
        tempLWordType: WordType = WordType.UNKNOWN
        tempCVExampleSentences: list = list()
        tempCVExampleSentencesAsString: str = ""
        tempCVLevenshteinToSiblings: list = list()
        tempCVLevenshteinToSiblingsAsString: str = ""
        tempCVLevenshteinDistanceToParent: int = 0
        tempDataRow = ""
        targetPath = targetDirectory + "\\" + fileName + ".csv"
        headerRow = ""

        #   Verify directory exists. If not, return false
        if not os.path.isdir(targetDirectory):
            SystemMessages.DirectoryNotFound(targetDirectory)
            return False
        
        #   Open the file for write
        writer = open(targetPath, mode="w", encoding="utf-8")

        #   Retrieve copy of the dictionary
        lDictCopy = self.__lemmaDictionary.copy()

        #   Calculate t_l_appearances
        for l in lDictCopy.values():
            totalLAppearances += l.GetOccurences()

        #   Update frequencies
        self.UpdateFrequencies()

        #   Compile and send header row
        for head in self.__outFileColumnHeaders:
            headerRow = headerRow + head + ','
        headerRow = headerRow[:-1]
        headerRow = headerRow + '\n'
        writer.write(headerRow)

        #   For each lemma and each cv in that lemma, write a datarow to the output file
        for l in lDictCopy.values():

            # Grab fields
            temp_l = l.GetValue()
            tempLemmaApearances = l.GetOccurences()
            tempPLemmaAppearances = l.GetRelativeFrequency()
            tempLLanguage = l.GetLanguage()
            tempLWordType = l.GetWordType()

            # For each cv, write a data row based on its lemma
            for cv in l.GetCVDictionary().values():

                # Grab fields
                temp_cv = cv.GetValue()
                tempCVAppearances = cv.GetOccurences()
                tempPCVAppearances = cv.GetRelativeFrequency()

                # Grab example sentences and format it for export
                tempCVExampleSentences = cv.GetExampleSentences()
                #if (debug):
                    #print(tempCVExampleSentences)
                if (len(tempCVExampleSentences) != 0):
                    tempCVExampleSentencesAsString = ";".join(tempCVExampleSentences)
                else:
                    tempCVExampleSentencesAsString = ""
                #if (debug):
                    #print(tempCVExampleSentencesAsString)

                # Grab levenshtein distance to siblings and format it for export
                tempDict = cv.GetLevenshteinDistanceToSiblingDictionary()
                tempCVLevenshteinToSiblings = []
                for key in tempDict.keys():
                    tempCVLevenshteinToSiblings.append("("+key+"|"+str(tempDict[key])+")")
                if len(tempCVLevenshteinToSiblings) != 0:
                    tempCVLevenshteinToSiblingsAsString = ";".join(tempCVLevenshteinToSiblings)
                    #tempCVLevenshteinToSiblingsAsString = ""
                else:
                    tempCVLevenshteinToSiblingsAsString = ""

                # Grab levenshtein distance to parent
                tempCVLevenshteinDistanceToParent = cv.GetLevenshteinDistanceToParent()

                # Compile data row
                #   Format: Lemma,L_appearances,L_language,L_word_type,Clean Value,CV_appearances,T_L_appearances,P_L_appearances,P_CV_appearances,Example Sentences, Levenshtein Distance to Siblings, Levenshtein Distance to Parent
                tempDataRow = ""
                tempDataRow += temp_l + ","
                tempDataRow += str(tempLemmaApearances) + ","
                tempDataRow += tempLLanguage.name + ","
                tempDataRow += tempLWordType.name + ","
                tempDataRow += temp_cv + ","
                tempDataRow += str(tempCVAppearances) + ","
                tempDataRow += str(totalLAppearances) + ","
                tempDataRow += str(tempPLemmaAppearances) + ","
                tempDataRow += str(tempPCVAppearances) + ","
                tempDataRow += tempCVExampleSentencesAsString.replace(",","") + ","
                tempDataRow += tempCVLevenshteinToSiblingsAsString.replace(",","") + ','
                tempDataRow += str(tempCVLevenshteinDistanceToParent)
                if (tempDataRow[len(tempDataRow)-1] != '\n'):
                    tempDataRow += '\n'

                if (debug):
                    if (temp_cv.__contains__(",")):
                        print(temp_cv)
                    if ("," in tempCVExampleSentencesAsString):
                        print(tempCVExampleSentencesAsString)

                # Write row
                writer.write(tempDataRow)
        
        # Close 
        writer.close()

        # Set the store path
        self.__storePath = targetPath

        # Initialize a new dataframe based on the file just created one isn't initialized already
        self.__lemmaDataFrame = pd.read_csv(self.__storePath)

        # Return
        return True

    def PartitionDF(self, percentages: list) -> list:
        # create an empty list with the appropriate amount of slots
        returnList = [[], [], [], []]
        df0Size = len(self.__lemmaDataFrame.index)
        # check that DF is present loaded etc
        # If the dictionary isn't already loaded, return false
        if (self.__loaded == False):
            return False

        # check that percentages list totals up to 100% ie 1.0
        percentageCheck = 0
        for i in percentages:
            if (type(i) == float):
                percentageCheck += i
            elif (type(i) == int):
                percentageCheck += (i / 100)
            else:
                return False
        if(percentageCheck != 1.0):
            # print("values inaccurate or do not add to 100%")
            return False
        
        # first element in returnList is the full, shuffled 
        returnList[0] = self.__lemmaDataFrame.copy().sample(frac = 1.0)

        # split dataframe and put each new dataframe into returnList:

        # first, make a list of the lemma objects
        lemmalist = []
        for lemma in self.__lemmaDictionary:
            lemmalist.append(lemma)

        # shuffle those
        np.random.shuffle(lemmalist)

        nextDF = 0
        counter = round(percentages[nextDF] * len(lemmalist))
        templist = []
        for lemma in lemmalist:
            # print(counter)
            # if statement counts down until current df being written to is filled, also catches errors
            # where rounding causes percents to not add up to 100% and there is a leftover lemma
            if (counter == 0): #and (nextDF != (len(percentages)))): 
                nextDF += 1
                counter = round(percentages[nextDF] * len(lemmalist))
                returnList[nextDF] = (list(templist))
                templist.clear()
            # get the lemma from the dictionary and add to a temp list
            templist.append(lemma)
            counter -= 1
        returnList[nextDF + 1] = (list(templist))


        # make actual dataframes
        dfslist = []
        dfslist.append(returnList[0])
        dfnameslist = []
        for i in range(len(percentages)):
            varname = "df" + str(round(percentages[i] * 100))
            dfnameslist.append(varname)
            correctlist = returnList[i + 1]
            tempdict = {}
            addtodf = []
            for lemmastring in correctlist:
                for cvstring in self.GetLemma(lemmastring).GetCVDictionary():
                    tempdict = {
                        "Lemma": lemmastring,
                        "L_appearances": self.GetLemma(lemmastring).GetOccurences(),
                        "L_language": self.GetLemma(lemmastring).GetLanguage(),
                        "L_word_type": self.GetLemma(lemmastring).GetWordType(),
                        "Clean Value": cvstring,
                        "CV_appearances": self.GetLemma(lemmastring).GetCV(cvstring).GetOccurences(),
                        "T_L_appearances": 305573,
                        "P_L_appearances":  "error", #self.GetLemma(lemmastring).GetOccurences()/305573,
                        "P_CV_appearances": "error", # self.GetLemma(lemmastring).GetCV(cvstring).GetOccurences()/305573,
                        "Example Sentences": self.GetLemma(lemmastring).GetExampleSentences(),
                        "Levenshtein Distance to Siblings": "error", #self.GetLemma(lemma).GetCV(cvstring).GetLevenshteinDistanceToSibling(cvstring)
                        "Levenshtein Distance to Parent": "error", #  self.GetLemma(lemmastring).GetCV(cvstring).GetLevenshteinDistanceToParent()
                    }
                    addtodf.append(tempdict)
                
            varname = pd.DataFrame(addtodf, columns=["Lemma","L_appearances","L_language","L_word_type","Clean Value","CV_appearances","T_L_appearances","P_L_appearances","P_CV_appearances","Example Sentences","Levenshtein Distance to Siblings","Levenshtein Distance to Parent"])
            dfslist.append(varname)

        joinstring = ", "
        print("Copied lemmaDataFrame at index 0 of return. Loaded dataframes: ", joinstring.join(dfnameslist), " beginning at index 1.")
        return dfslist

    """
      This is your general query method to get different lemmas by different filters
          and orders.
    
         Expects a list of filters in SQL style. 
          When thinking of SQL, we filter using
              WHERE (column and condition) AND (column condition) ... etc
              and example might be
              WHERE col1 < 10 AND col2 = 'pizza'
              here we would input: ['col1 < 10', 'col2 = "pizza"']
              as the filters list
          source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html
    
          When thinking of SQL, we order using column names
              ORDERBY col1, col2 
              here the orderby list would appear as: ['col1','col2']
          source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html

    Notes on Querying:
        Querying is similar to SQL, but a little different. I will provide how to
        use filters that most likely will occur.

    Column Names and their use:
        Lemma (str): The name of the lemma associated with the data in the row
        L_appearances (int): The number of appearances the lemma has in the dataset
        L_language (str): The language of the lemma
        L_word_type (str): The word type of the lemma (noun, verb, etc.)
        Clean Value (str): The name of the clean value associated with the lemma in the row
        CV_appearances (int): The number of appearances the clean value has in the dataset when associated with the row's lemma
        T_L_appearances (int): The total number of all lemma appearances in the dataset (constant for all rows).
        P_L_appearances (float): The percent appearance of the row's Lemma in relation to the total number of lemmas (L_appearances/T_L_appearances)
        P_CV_appearances (float): The percent appearance of the row's Clean Value in relation to the total number of clean value appearances associated with the lemma (CV_appearances/L_appearances)
        Example Sentences (str): Examples of CV being used in a sentence. Format: [sentence1: str; sentence2: str; ...]
        Levenshtein Distance to Siblings (str): Levenshtein distance to siblings as token pairs in a list. Format: [(siblingCV1: str| distance: int); (siblingCV2: str| distance: int); ...]
        Levenshtein Distance to Parent (int): Levenshtein distance to parent.    
 
    Of course, when querying, the format and datatypes of the column are important, but this is only for filtering and ordering.
    When using the Lemmas the query produces, you can do so defined by the Lemma class in code (for complex things like Levenshtein distance, it looks ugly in 
        a datarow, but you only need to access it via the Lemma's method GetCV() followed by GetLevenshteinDistanceDict() to get the levenshtein distances for
        example)

    When referencing a column, the query assumes what you enter that isn't a number is a column name.
    If there is confusing or it is throwing errors, add the back tick `` on both sides of the column name for further specification.

    For filters:
        When using a number (float or int), the query is smart enough that typing in the number is good enough
        When using a string, the value MUST have quotes around it (if your string is defined by single quotes, use double quotes and vice versa)

        Here are some general filters you may use when filtering:
        (col1 and col2 refer to example column names)

        Mathematical
            Type the condition as you would in Python in an if statement, using the column names as the variable to compare the number to
            ex.
                =: col1 == 100 
                <: col1 < 100
                >=: col1 >= 100
        
        Text
            Type the condition as you would in python. The column name can be used as a variable.
            ex.
                =: col1 == "pattern"
            For LIKE or CONTAINS, use the following syntax
                LIKE: col1.str.contains("pattern")

        An example list of filters might be:
            filters = ['L_appearances >= 1000', 'Lemma.str.contains("ina")', 'CV_appearances < 2']

    For orderby:
        Input the column names as strings in a list. The list outputted will be orderedBy the first value you input, then the second, and so on.

        Ex.
        orderBy=['col2', 'col1'])
        Before      col1  col2  col3 col4      After    col1  col2  col3 col4
                    1    B     1     1    B             0    A     2     0    a         
                    0    A     2     0    a             1    B     1     1    B
                    2    C     9     9    c             2    C     9     9    c
                    5    F     4     3    F             3    D     8     4    D
                    4    E     7     2    e             4    E     7     2    e
                    3    D     8     4    D             5    F     4     3    F            
                    
        Hence if the lemmas were A,B,C,D,E, and F, you would recieve the lemmas back in alphabetical order.


    Other more relevant examples of filtering:
        For getting lemmas by most frequent and only if it has 1000 or more appearances: lemmaDictionary.LemmaQuery(['L_appearances >= 1000'],['L_appearances'])
        For getting lemmas with the SUMERIAN language: lemmaDictionary.LemmaQuery(['L_language.str.contains("SUMERIAN")'],[])
        For getting lemmas infrequent ( < 100) in AKKADIAN: lemmaDictionary.LemmaQuery(['L_language.str.contains("AKKADIAN")','L_appearances < 100'],[])

    Run this with any of the queries in main to test!
    pickup = lemmaDictionary.LemmaQuery(['L_language.str.contains("AKKADIAN")','L_appearances < 100'],[])
    for lemma in pickup:
        print(lemma.GetValue() + ": " + str(lemma.GetOccurences()))
    """

    def LemmaQuery(self, filters: list = [], orderBy: list = [], selectDF: DataFrame = None, debug=False) -> list:

        # Locals
        queryDF = None
        returnList = list()
        uniqueLemmaDict = dict()

        # If store hasn't been ran yet (there isn't a store path), there isn't a dataframe to query from. Therefore, return.
        if (self.__storePath == None):
            SystemMessages.LD_InvalidQueryStoreNotRan()
            return list()
            # checking that the dataframe is not empty and is not returning False, which happens when the percentages 

        # Initialize the queryDF as a copy of the lemmaDataFrame if using 100% of original dataframe
        # else, the DF is passed in by indexing the list that PartitionDF returns
        if (selectDF is None):
            queryDF = self.__lemmaDataFrame.copy()
        else:
            queryDF = selectDF


        # For each filter, run a query and filter the queryDF based on that filter
        for i in range(0,len(filters)):
            queryDF = queryDF.query(filters[i])

        # Sort values by "orderby" specifications
        if (orderBy != []):
            queryDF = queryDF.sort_values(orderBy, ascending=False)
        if debug:
            print(queryDF)

        # For each of the returned rows with unique lemmas, look for them in the lemma dictionary
        #   in the order they appear in the dataframe
        #   add them to a list to return.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for i in range(0,len(queryDF)):
                uniqueLemmaDict[queryDF.iloc[i][0]] = self.__lemmaDictionary[queryDF.iloc[i][0]]

        # Convert unique dictionary to list and return
        #   This implementation works since values are inserted into dictionary by appearance and stay in that order
        #   therefore the dictionary values maintain the order from "orderby" since the rows were parsed from top
        #   to bottom.
        returnList = list(uniqueLemmaDict.values())
        return returnList


    """
        Used to load the pandas dataframe used when querying in the lemmaDictionary
        from the dictionary's stored location.
    """
    def __LoadQueryDataFrame(self):
        if (self.__storePath != ""):
            
            # Tokenize store path
            storePathTokens = self.__storePath.split("\\")
            endingToken = storePathTokens[len(storePathTokens)-1]
            storePathTokens.pop()

            # Store Dictionary as queriable dataset. Then query the dataset (in store method)
            #   This method helps reduce the chance of race condition.
            self.StoreDictionary("\\".join(storePathTokens), endingToken, False)

    #
    #   Used in testing   
    #
    """
    # 
    #   Used to log a test case for a CV found within this lemma.
    #   Enter the lemma's name, CV's name, if the guess was correct, and optionally what the incorrectly guessed word was.
    #   The test data is then stored within the lemma's CV if found within its CV dictionary and the lemma is found within
    #       the lemma dictionary.
    #
    """
    def LogTest(self, lemma: str, cv: str, correct: bool, incorrectGuessedWord: str = None):
        # Insert the test into the this dictionary in the lemma's corresponding lemma object if found within the dictionary.
        if (self.__lemmaDictionary.__contains__(lemma)):
            self.__lemmaDictionary[lemma].LogTest(cv, correct, incorrectGuessedWord)
        else:
            SystemMessages.LD_LNotFound(lemma)

    """
        Used to reset all lemma test variables in the lemma dictionary to their default values.
        For Lemmas
            Tested -> False
        For CVs
            Tested -> False
            CorrectGuessesCount -> 0
            IncorrectGuessCount -> 0
            IncorrectGuessWords -> []
    """
    def FlushTests(self):
        for l in self.__lemmaDictionary.values():
            l.ResetTests()
        #self.__LoadQueryDataFrame() # Unneeded as values in queriedDataFrame access dictionary values which are flushed, but the dataframe doesn't hold test data by reference.
        self.__totalCorrectTestCount = 0
        self.__totalTestCount = 0

    """
    Given a directory name and file name at the minimum,
    This method stores test data from all this object instance's lemmas and cvs used in testing in a .csv file
        with the name specified in "fileName."
     
    Outputs data in the form: (Brackets imply optional data contained within and will be removed if unspecified)

        |[testName]                     |                       |                                    |                          |                              |
        | Statistical Analysis:         |                       |                                    |                          |                              |
        |totalGuesses                   | (int)                 |                                    |                          |                              |
        |totalCorrectGuesses            | (int)                 |   percentCorrectGuesses            |   (float)                |                              |
        |totalIncorrectGuesses          | (int)                 |   percentIncorrectGuesses          |   (float)                |                              |
        |totalAkkadianGuesses           | (int)                 |                                    |                          |                              |   
        |totalCorrectAkkadianGuesses    | (int)                 |   percentCorrectAkkadianGuesses    |   (float)                |                              |
        |totalIncorrectAkkadianGuesses  | (int)                 |   percentIncorrectAkkadianGuesses  |   (float)                |                              |
        |totalSumerianGuesses           | (int)                 |                                    |                          |                              |
        |totalCorrectSumerianGuesses    | (int)                 |   percentCorrectSumerianGuesses    |   (float)                |                              |
        |totalIncorrectSumerianGuesses  | (int)                 |   percentIncorrectSumerianGuesses  |   (float)                |                              |
        |totalEmesalGuesses             | (int)                 |                                    |                          |                              |                                   
        |totalCorrectEmesalGuesses      | (int)                 |   percentCorrectEmesalGuesses      |   (float)                |                              |
        |totalIncorrectEmesalGuesses    | (int)                 |   percentIncorrectEmesalGuesses    |   (float)                |                              |
        |                               |                       |                                    |                          |                              |
        | Qualitative Analysis:         |                       |                                    |                          |                              | 
        |Model Used                     | [modelName]           |                                    |                          |                              | 
        |Prompts Used                   | [promptsUsed]         |                                    |                          |                              | 
        |                               |                       |                                    |                          |                              | 
        | Tested Lemmas:                |                       | Total Tests                        | Percent Correct          | Percent Incorrect            |
        | (str) (lemma1 name)           |                       | (int) (totalTests)                 | (float) (percentCorrect) | (float) (percentIncorrect)   | 
        |                               | WordType: (str)       |                                    |                          |                              |
        |                               | Language: (str)       |                                    |                          |                              | 
        |                               | (str) (lemma1 cv1)    | (int) (totalTestsOfCV)             | (float) (percentCorrect) | (float) (percentIncorrect)   | 
        |                               | (str) (lemma1 cv2)    | (int) (totalTestsOfCV)             | (float) (percentCorrect) | (float) (percentIncorrect)   | 
        |                               | (str) (lemma1 cv3)    | (int) (totalTestsOfCV)             | (float) (percentCorrect) | (float) (percentIncorrect)   | 
        |                               |  .                    |                                    |                          |                              | 
        |                               |  .                    |                                    |                          |                              | 
        |                               | (str) (lemma1 cvn)    | (int) (totalTestsOfCV)             | (float) (percentCorrect) | (float) (percentIncorrect)   | 
        | (str) (lemma2 name)           |                       | (int) (totalTests)                 | (float) (percentCorrect) | (float) (percentIncorrect)   | 
        |                               | WordType: (str)       |                                    |                          |                              |
        |                               | Language: (str)       |                                    |                          |                              | 
        |                               | (str) (lemma1 cv1)    | (int) (totalTestsOfCV)             | (float) (percentCorrect) | (float) (percentIncorrect)   | 
        |                               | (str) (lemma1 cv2)    | (int) (totalTestsOfCV)             | (float) (percentCorrect) | (float) (percentIncorrect)   | 
        |                               | (str) (lemma1 cv3)    | (int) (totalTestsOfCV)             | (float) (percentCorrect) | (float) (percentIncorrect)   | 
        |                               | (str) (lemma2 cv1)    | (int) (totalTestsOfCV)             | (float) (percentCorrect) | (float) (percentIncorrect)   | 
                                            .
                                            .
                                            .
                                            etc.
                                            
    Optional:
        testName: A name for the test contained within the lemma dictionary.
        promptsUsed: A list of prompts used in testing. Stored at the top of the .csv file.
        modelName: The name of the model used in testing. Stored at the top of the .csv file.
        debug: A flag on whether to print messages while this method executes (true) or not (false). Used in debugging.
    Returns:
        True if successfully stored
        False otherwise
    """
    def StoreTestData(
            self, 
            targetDirectory: str, 
            fileName: str, 
            testName: str = "n/a",
            description: str = "n/a",
            promptsUsed: list = [],
            modelName: str = "n/a",
            debug: bool=False
        ) -> bool:

        # Locals
        writer = None
        lDictCopy: dict = dict()
        targetPath = targetDirectory + "\\" + fileName + ".csv"
        totalGuesses: int = 0
        totalCorrectGuesses: int = 0
        totalAkkadianGuesses: int = 0
        totalCorrectAkkadianGuesses: int = 0
        totalSumerianGuesses: int = 0
        totalCorrectSumerianGuesses: int = 0        
        totalEmesalGuesses: int = 0
        totalCorrectEmesalGuesses: int = 0
        tempInt1: int = 0
        tempInt2: int = 0 
        tempInt3: int = 0
        tempInt4: int = 0
        lemmaTotalGuessDict: dict = dict()
        lemmaTotalCorrectGuessDict: dict = dict()


        # Verify directory's existance.
        if not os.path.isdir(targetDirectory):
            SystemMessages.DirectoryNotFound(targetDirectory)
            return False
        
        #   Open the file for write
        writer = open(targetPath, mode="w", encoding="utf-8")

        #   Get a copy of the dictionary
        lDictCopy = self.__lemmaDictionary.copy()

        # Begin writing
        if (debug):
            print("Begin writing test data")
        # Write header
        writer.write(testName + '\n')

        # Write statistical data
        #   Gather statistical data (note we make 2 passes through the dictionary to be memory efficient instead of time efficient.)
        for l in lDictCopy.values():
            if l.Tested():
                tempInt1 = l.TestCorrectGuessCount()
                tempInt2 = l.TestTotalGuessCount()
                if (l.GetLanguage() == Language.AKKADIAN):
                    totalAkkadianGuesses += tempInt2
                    totalCorrectAkkadianGuesses += tempInt1
                elif (l.GetLanguage() == Language.SUMERIAN):
                    totalSumerianGuesses += tempInt2
                    totalCorrectSumerianGuesses += tempInt1
                elif (l.GetLanguage() == Language.EMESAL):
                    totalEmesalGuesses += tempInt2
                    totalCorrectEmesalGuesses += tempInt1
                    
        #   Write overall statistical analysis
        writer.write("Statistical analysis:\n")
        writer.write("Total Guesses:,"+str(self.__totalTestCount)+'\n')
        if (totalGuesses != 0):
            writer.write("Total Correct Guesses:,"+str(self.__totalTestCount)+','+"Percent Correct Guesses:," + str(self.__totalCorrectTestCount/self.__totalTestCount)+'\n')
            writer.write("Total Incorrect Guesses:,"+str(self.__totalTestCount-self.__totalCorrectTestCount)+','+"Percent Incorrect Guesses:," + str((self.__totalTestCount-self.__totalCorrectTestCount)/self.__totalTestCount)+'\n')
        writer.write("Total Akkadian Guesses:,"+str(totalAkkadianGuesses)+'\n')
        if (totalAkkadianGuesses != 0):
            writer.write("Total Correct Akkadian Guesses:,"+str(totalCorrectAkkadianGuesses)+','+"Percent Correct Guesses:," + str(totalCorrectAkkadianGuesses/totalAkkadianGuesses)+'\n')
            writer.write("Total Incorrect Akkadian Guesses:,"+str(totalAkkadianGuesses-totalCorrectAkkadianGuesses)+','+"Percent Incorrect Guesses:," + str((totalAkkadianGuesses-totalCorrectAkkadianGuesses)/totalAkkadianGuesses)+'\n')
        writer.write("Total Sumerian Guesses:,"+str(totalSumerianGuesses)+'\n')
        if (totalSumerianGuesses != 0):
            writer.write("Total Correct Sumerian Guesses:,"+str(totalCorrectSumerianGuesses)+','+"Percent Correct Guesses:," + str(totalCorrectSumerianGuesses/totalSumerianGuesses)+'\n')
            writer.write("Total Incorrect Sumerian Guesses:,"+str(totalSumerianGuesses-totalCorrectSumerianGuesses)+','+"Percent Incorrect Guesses:," + str((totalSumerianGuesses-totalCorrectSumerianGuesses)/totalSumerianGuesses)+'\n')
        writer.write("Total Emesal Guesses:,"+str(totalEmesalGuesses)+'\n')
        if (totalEmesalGuesses != 0):
            writer.write("Total Correct Guesses:,"+str(totalCorrectEmesalGuesses)+','+"Percent Correct Guesses:," + str(totalCorrectEmesalGuesses/totalEmesalGuesses)+'\n')
            writer.write("Total Incorrect Guesses:,"+str(totalEmesalGuesses-totalCorrectEmesalGuesses)+','+"Percent Incorrect Guesses:," + str((totalEmesalGuesses-totalCorrectEmesalGuesses)/totalEmesalGuesses)+'\n')
        writer.write('\n')

        #   Write qualiatiative analysis
        writer.write("Qualitative analysis:\n")
        writer.write("Description:,"+description+'\n')
        writer.write("Model used:,"+modelName+'\n')
        writer.write("Prompts used:,\n")
        for prompt in promptsUsed:
            writer.write(str(prompt) + '\n')
        writer.write('\n')


        #   Write statistical data on each individual lemma
        writer.write("Tested Lemmas:,Total Tests,,Percent Correct,Percent Incorrect,Incorrect Guesses\n")
        for l in lDictCopy.values():
            if (l.Tested()):
                writer.write(l.GetValue()+','+str(l.TestTotalGuessCount())+','+str(l.TestCorrectGuessCount()/l.TestTotalGuessCount())+','+str(l.TestIncorrectGuessCount()/l.TestTotalGuessCount())+'\n')
                writer.write(',WordType: ' + l.GetWordType().name+'\n')
                writer.write(',Language: ' + l.GetLanguage().name+'\n')
                writer.write(',Tested Clean Values\n')
                for cv in l.GetCVDictionary().values():
                    if (cv.Tested()):
                        temp1 = cv.TestCorrectGuessCount()
                        temp2 = cv.TestIncorrectGuessCount()
                        writer.write(','+cv.GetValue()+','+str(temp1+temp2)+','+str(temp1/(temp1+temp2))+','+str(temp2/(temp1+temp2))+',')
                        if (len(cv.TestIncorrectGuessWords()) > 0):
                            writer.write(" | ".join(cv.TestIncorrectGuessWords()))
                        writer.write('\n')


        # Close and return when finished
        writer.close()
        return True

    """
        Used to increment correct/total guesses in this object.
    """
    def __IncrementTotalTestCount(self):
        self.__totalTestCount+=1
    def __IncrementTotalCorrectTestCount(self):
        self.__totalCorrectTestCount+=1

    """
    #   Sorting algorithm that sorts Lemmas by frequency in descending order (From CV in DataHelpers.py)
    """
    def __LemmaFrequencyQuickSort(cvList: list) -> list:
        LemmaDictionary.__LemmaFrequencyQuickSortHelper(cvList, 0, len(cvList)-1)
        return cvList
    def __LemmaFrequencyQuickSortHelper(lemmaList: list, low: int, high: int) -> list:
        if low < high:
            pivot: int = LemmaDictionary.__partition(lemmaList, low, high)
            LemmaDictionary.__LemmaFrequencyQuickSortHelper(lemmaList, low, pivot-1)
            LemmaDictionary.__LemmaFrequencyQuickSortHelper(lemmaList, pivot+1, high)
    def __partition(lemmaList: list,  low: int, high: int) -> int:
        pivot: int = lemmaList[high]
        i: int = low -1
        j: int
        tempLemma: Lemma
        for j in range(low , high):
            if lemmaList[j].GetOccurences() >= pivot.GetOccurences():
                i+=1
                tempLemma = lemmaList[i]
                lemmaList[i] = lemmaList[j]
                lemmaList[j] = tempLemma
        tempLemma = lemmaList[i+1]
        lemmaList[i+1] = lemmaList[high]
        lemmaList[high] = tempLemma
        return i+1


    """
    #   Gets each Lemma in the lemma dictionary that has the inputted word type
    #   (depreciated. Use LemmaDictionary query())
    """  
    def GetLemmasByWordType(self, targetType: WordType) -> list:
        rl = []
        for l in self.__lemmaDictionary.values():
            if l.GetWordType() == targetType:
                rl.append(l)
        return rl

    """
    #   Gets each Lemma in the lemma dictionary that has the inputted language type
    #   (depreciated. Use LemmaDictionary query())
    """
    def GetLemmasByLanguage(self, targetType: Language) -> list:
        rl = []
        for l in self.__lemmaDictionary.values():
            if l.GetLanguage() == targetType:
                rl.append(l)
        return rl
    
    """
    #   Gets each lemma in the lemma dictionary ordered by frequency of appearance
    #   in decending order (most frequent first)
    #   (depreciated. Use LemmaDictionary query())
    """
    def GetLemmasByFrequency(self) -> list:
        return LemmaDictionary.__LemmaFrequencyQuickSort(list(self.__lemmaDictionary.values()))


#
#   Class focused on retrieval of, storage of, and access to clean value TESTING data
#
TSSentenceMaskToken: str = CVExampleSentenceMaskToken
class TestSystem:



    #
    #   Constants
    #
    #   Related to test file
    __columnFragmentID: Final[str] = "fragment_id"
    __columnFragmentLineNum: Final[str] = "fragment_line_num"
    __columnIndexInLine: Final[str] = "index_in_line"
    __columnWordLanguage: Final[str] = "word_language"
    __columnValue: Final[str] = "value"
    __columnPrediction: Final[str] = "prediction"
    #   Related to packets
    __packetFragmentID: Final[str] = "fragmentID"
    __packetSentence: Final[str] = "sentence"
    __packetSentenceNumber: Final[str] = "sentenceNumber"
    __packetSentenceTokens: Final[str] = "sentenceTokens"
    __packetLanguage: Final[str] = "language"
    __packetValues: Final[str] = "values"
    __packetPredictions: Final[str] = "predictions"
    __packetCleanValues: Final[str] = "cleanValues"

    #
    #   Fields
    #
    __currentPacket: dict  
    __nextPacket: dict 
    __streamReader = None
    __streamWriter = None
    __endOfReader: bool
    __columnHeadIndexes: dict # Marks where each column is as columns are read in from the stream reader as well as how packets are sent to the streamwriter


    #
    #   Constructor
    #
    def __init__(self):

        # Init locals
        self.__endOfReader = True
        self.__columnHeadIndexes = { # Marks where each column is as columns are read in from the stream reader as well as how packets are sent to the streamwriter
            self.__columnFragmentID : 0,
            self.__columnFragmentLineNum : 1,
            self.__columnIndexInLine : 2,
            self.__columnWordLanguage : 3,
            self.__columnValue : 4,
            self.__columnPrediction : 5
        }

        self.__currentPacket = dict()
        self.__nextPacket = dict()


    #
    #   Methods
    #
    """
        Returns the current packet the testsystem is holding
    """
    def GetCurrentPacket(self) -> dict:
        return self.__currentPacket


    """
        Loads test file
    """
    def LoadTestFile(self, filePath: str) -> bool:


        #   Locals
        streamReader = None
        currentRow: str = ""
        currentRowTokens: str = ""

        # Attempt to open the file from the path provided.
        #   Return false if file cannot be opened/isn't found
        if (not os.path.exists(filePath)):
            SystemMessages.FileNotFound(filePath)
            return False
        else:
            streamReader = open(filePath, "r", encoding="utf8")

        # Read first line. If empty, return
        currentRow = streamReader.readline()

        if len(currentRow) == 0:
            streamReader.close()
            SystemMessages.TS_InvalidFileFormat(filePath)
            return False
        # else:
            # *Depreciated
            # currentRowTokens = currentRow.split(',')
            # print(currentRow)
            # print(str(currentRowTokens))
            # # Reset column head indexes
            # for key in list(self.__columnHeadIndexes.keys()):
            #     self.__columnHeadIndexes[key] = -1

            # # Determine where column heads are in the file (assumed to be constant)
            # for i in range(0, len(currentRowTokens)):
            #     keyList = list(self.__columnHeadIndexes.keys())
            #     for j in keyList:
            #         if currentRowTokens[i] in j:
            #             self.__columnHeadIndexes[j] = i
            

        # If any of the columnHeadIndexes are missing (still set to -1), return false as not all column heads are present.
        # if -1 in list(self.__columnHeadIndexes.values()):
        #     SystemMessages.TS_InvalidFileFormat(filePath)
        #     print(str(self.__columnHeadIndexes))
        #     streamReader.close()
        #     return False
        
        # Otherwise, set this object's stream reader to the retrieved stream reader.
        self.__streamReader = streamReader
        self.__endOfReader = False

        # Return when finished
        return True

    """
        Generates an empty packet ready for use
    """
    def GenerateBlankPacket(self) -> dict:
        returnDict = {
            self.__packetFragmentID : -1,
            self.__packetLanguage: Language.UNKNOWN,
            self.__packetSentence: "",
            self.__packetSentenceTokens: list(),
            self.__packetSentenceNumber: -1,
            self.__packetValues: list(),
            self.__packetCleanValues: list(),
            self.__packetPredictions : list()
        }
        return returnDict

    """
        Given the test dictionary is open for reading, 
        This method gets the next sentence in the file along with their cvs order they appear.
        Returns them in the format of:
        dict = {
            sentence: [str],
            sentenceNumber: [int],
            sentenceTokens: [list],
            fragmentLineNum: [int],
            fragmentID: [int],
            language: [Language],
            values: [
                [cv1: str],
                [cv2: str],
                [cv3: str],
                [cv4: str],
                .
                .
                .
                [cvn: str]
            ]
            predictions: [
                [cv1 lemma guess: str],
                
            ]
        }

        If the file is not open or if the file has reached the end, a blank dictionary is returned.
    """
    def NextPacket(self) -> dict:
        
        # Locals
        dirty: bool = False
        currentRow: str = ""
        currentRowTokens: list = []
        tempSentenceNumber: int = -1
        tempFragmentID: int = -1
        tempLanguage: Language = Language.UNKNOWN
        tempValue: str = ""
        tempIndexInLine: int = -1


        # If the reader is at the end of the file or hasnt been initialized, return a blank dictionary.
        if (self.__endOfReader == True):
            SystemMessages.TS_NoPackets()
            return dict()

        # Read the next packet
        #   If next packet is initialized, gather its information and set it to the current packet. 
        #   Otherwise, start a new packet
        if (self.__nextPacket != {}):
            self.__currentPacket = self.__nextPacket
            self.__nextPacket = dict()
            dirty = True
        else:
            self.__currentPacket = self.GenerateBlankPacket()

        # Read the first line, 
        currentRow = self.__streamReader.readline()

        # If empty and not dirty, return a blank packet. End of file = true
        if (currentRow == "" and dirty == False):
            self.__currentPacket = dict()
            self.__endOfReader = True
            self.__streamReader.close()
            return self.__currentPacket
        # If empty and dirty, return the current packet. End of file = true
        elif (currentRow == ""):
            self.__endOfReader = True
            self.__streamReader.close()
            return self.__currentPacket

        # Parse the packet. Gather its tokens. Set the current packet only if not dirty
        if not dirty:

            currentRowTokens = currentRow.split(",")
            for i in range(0, len(currentRowTokens)):

                # Fragment id case
                if self.__columnHeadIndexes[self.__columnFragmentID] == i:
                    tempFragmentID = int(currentRowTokens[i])

                # Fragment line num case
                elif self.__columnHeadIndexes[self.__columnFragmentLineNum] == i:
                    tempSentenceNumber = int(currentRowTokens[i])

                # Index in line case
                elif self.__columnHeadIndexes[self.__columnIndexInLine] == i:
                    tempIndexInLine = int(currentRowTokens[i])

                # Language token case
                elif self.__columnHeadIndexes[self.__columnWordLanguage] == i:
                    match (currentRowTokens[i]):
                        case "AKKADIAN":
                            tempLanguage = Language.AKKADIAN
                        case "EMESAL":
                            tempLanguage = Language.EMESAL
                        case "SUMERIAN":
                            tempLanguage = Language.SUMERIAN

                # Value case
                elif self.__columnHeadIndexes[self.__columnValue] == i:

                    # Set value
                    tempValue = currentRowTokens[i]

            # End for

            # Set up current packet based on inputs 
            self.__currentPacket[self.__packetFragmentID] = tempFragmentID
            self.__currentPacket[self.__packetSentenceNumber] = tempSentenceNumber
            self.__currentPacket[self.__packetLanguage] = tempLanguage
            self.__currentPacket[self.__packetValues].append(tempValue)
            self.__currentPacket[self.__packetCleanValues].append(tempValue.replace('-',''))

            #   Append to the end of the example sentence the CV found
            #       If the IndexInLine of the row is greater than the number of sentence tokens,
            #       i.e. there are missing words in the sentence between the previous row and the
            #       current row, add mask tokens inbetween
            self.__currentPacket[self.__packetSentenceTokens] = list()
            while (int(tempIndexInLine) - 1 > len(self.__currentPacket[self.__packetSentenceTokens])):
                self.__currentPacket[self.__packetSentenceTokens].append(TSSentenceMaskToken)
            self.__currentPacket[self.__packetSentenceTokens].append(tempValue)

            # Read the next row
            currentRow = self.__streamReader.readline()
                  
        # Read the next line, until the end of the file or until the sentence numbers don't match.
        #   In either case, complete the current packet and return it.
        while not self.__endOfReader: 

            # Reset locals
            tempFragmentID = -1
            tempSentenceNumber = -1
            tempIndexInLine = -1
            tempLanguage = Language.UNKNOWN
            tempValue = ""

            # If end of file, return
            if (currentRow == ""):

                # Finish the sentence and return
                self.__currentPacket[self.__packetSentence] = (' '.join(self.__currentPacket[self.__packetSentenceTokens]) + '.').replace('-','')

                self.__endOfReader = True
                self.__streamReader.close()
                return self.__currentPacket        

            # Otherwise, gather the tokens
            else:

                currentRowTokens = currentRow.split(",")
                for i in range(0, len(currentRowTokens)):

                    # Fragment id case
                    if self.__columnHeadIndexes[self.__columnFragmentID] == i:
                        tempFragmentID = int(currentRowTokens[i])

                    # Fragment line num case
                    elif self.__columnHeadIndexes[self.__columnFragmentLineNum] == i:
                        tempSentenceNumber = int(currentRowTokens[i])

                    # Index in line case
                    elif self.__columnHeadIndexes[self.__columnIndexInLine] == i:
                        tempIndexInLine = int(currentRowTokens[i])

                    # Language token case
                    elif self.__columnHeadIndexes[self.__columnWordLanguage] == i:
                        match (currentRowTokens[i]):
                            case "AKKADIAN":
                                tempLanguage = Language.AKKADIAN
                            case "EMESAL":
                                tempLanguage = Language.EMESAL
                            case "SUMERIAN":
                                tempLanguage = Language.SUMERIAN

                    # Value case
                    elif self.__columnHeadIndexes[self.__columnValue] == i:

                        # Set value
                        tempValue = currentRowTokens[i]

                # Check the sentence number.
                #   If it's different than the current packet's sentence number, complete the current packet, start the next packet
                #       and return the current packet.
                #   Otherwise, add the row's contents to the current packet
                if (tempSentenceNumber != self.__currentPacket[self.__packetSentenceNumber]):

                    # Complete the current packet
                    self.__currentPacket[self.__packetSentence] = (' '.join(self.__currentPacket[self.__packetSentenceTokens]) + '.').replace('-','')

                    # Assemble the next sentence using the fields gathered.
                    self.__nextPacket = self.GenerateBlankPacket()
                    self.__nextPacket[self.__packetFragmentID] = tempFragmentID
                    self.__nextPacket[self.__packetSentenceNumber] = tempSentenceNumber
                    self.__nextPacket[self.__packetLanguage] = tempLanguage
                    self.__nextPacket[self.__packetValues].append(tempValue)
                    self.__nextPacket[self.__packetCleanValues].append(tempValue.replace('-',''))

                    while (int(tempIndexInLine) - 1 > len(self.__nextPacket[self.__packetSentenceTokens])):
                        self.__nextPacket[self.__packetSentenceTokens].append(TSSentenceMaskToken)
                    self.__nextPacket[self.__packetSentenceTokens].append(tempValue)

                    # Return the current packet
                    return self.__currentPacket
                    
                else:

                    # Add content to current packet
                    self.__currentPacket[self.__packetValues].append(tempValue)
                    self.__currentPacket[self.__packetCleanValues].append(tempValue.replace('-',''))
                    while (int(tempIndexInLine) - 1 > len(self.__currentPacket[self.__packetSentenceTokens])):
                        self.__currentPacket[self.__packetSentenceTokens].append(TSSentenceMaskToken)
                    self.__currentPacket[self.__packetSentenceTokens].append(tempValue)      

                    # Read the next row
                    currentRow = self.__streamReader.readline()        


        return


    """
        Sends the inputted packet to the inputted filePath in mode write by default or append by preference

        Packet assumed to be in form of:
        dict = {
            sentence: [str],
            sentenceNumber: [int],
            sentenceTokens: [list],
            fragmentLineNum: [int],
            fragmentID: [int],
            language: [Language],
            values: [
                [cv1: str],
                [cv2: str],
                [cv3: str],
                [cv4: str],
                .
                .
                .
                [cvn: str]
            ]
            predictions: [
                [cv1 lemma guess: str],
                
            ]
        }

    """
    def SendPacket(self, predictionPacket: dict, targetDirectory: str, fileName: str, append: bool = False):
                    
        # Locals
        streamWriter = None
        outputRow: str = ""
        targetPath = targetDirectory + "\\" + fileName + ".csv"


        #   Verify directory exists. If not, return false
        if not os.path.isdir(targetDirectory):
            SystemMessages.DirectoryNotFound(targetDirectory)
            return False
        
        #   Open the file for write if append == false
        if (append == False):
            streamWriter = open(targetPath, mode="w", encoding="utf-8")
            outputRow = ""
            outputRow += self.__columnFragmentID + ','
            outputRow += self.__columnFragmentLineNum + ','
            outputRow += self.__columnIndexInLine + ','
            outputRow += self.__columnWordLanguage + ','
            outputRow += self.__columnValue + ','
            outputRow += self.__columnPrediction + '\n'
            streamWriter.write(outputRow)

        else:
            streamWriter = open(targetPath, mode="a", encoding="utf-8")
            
            # Get the size of the file to write to. If the size is 0, write the header columns
            if (os.stat(targetPath).st_size == 0):
                outputRow = ""
                outputRow += self.__columnFragmentID + ','
                outputRow += self.__columnFragmentLineNum + ','
                outputRow += self.__columnIndexInLine + ','
                outputRow += self.__columnWordLanguage + ','
                outputRow += self.__columnValue + ','
                outputRow += self.__columnPrediction + '\n'
                streamWriter.write(outputRow)


        #   For each sentence token in the packet, write the value output
        counter: int = 0
        for i in range(0, len(predictionPacket[self.__packetSentenceTokens])):
            
            if (predictionPacket[self.__packetSentenceTokens][i] != TSSentenceMaskToken):
                outputRow = ""
                outputRow += str(predictionPacket[self.__packetFragmentID]) + ','
                outputRow += str(predictionPacket[self.__packetSentenceNumber]) + ','
                outputRow += str(i+1) + ','
                outputRow += predictionPacket[self.__packetLanguage].name + ','
                outputRow += predictionPacket[self.__packetValues][counter] + ','
                
                # Check to make sure all predictions are present. Only write if a prediction is made
                if (len(predictionPacket[self.__packetPredictions]) > counter):
                    outputRow += predictionPacket[self.__packetPredictions][counter] + '\n'
                else:
                    outputRow += '\n'

                # Write the row
                streamWriter.write(outputRow)

                counter = counter + 1

        # Close the stream writer when done
        streamWriter.close()

        return



    

#
#   Class focused on retrieval of, storage of, and access to prompt data
#
class PromptDictionary:
    
    #
    #   Data Design Tokens
    #       Used for defining what to look for when loading Prompts and what they mean
    #
    __RulesToken: str = "Rules" # The token marking the beginning of the rules section which defines symbols and their meaning
    __PromptsToken: str = "Prompts" # The token marking the beginning of the prompts section which defines each prompt, their id, and type

    # The dictionary marking a rule and its corresponding symbol for prompt data to be
    #   ingested.
    __RulesRole: str = "Role"                                   #   PromptType.ROLE
    __RulesInstruction: str = "Instruction"                     #   PromptType.INSTRUCTION
    __RulesTrainingPrompt: str = "Training_Prompt"              #   PromptType.TRAINING_PROMPT
    __RulesTestingPrompt: str = "Testing_Prompt"                #   PromptType.TESTING_PROMPT
    __RulesConfirmationPostive: str = "Confirmation_Positive"   #   PromptType.CONFIRMATION_POSTITIVE
    __RulesConfirmationNegative: str = "Confirmation_Negative"  #   PromptType.CONFIRMATION_NEGATIVE
    __RulesParam: str = "Param"                                 #   The token in the rules section marking the param symbol in each prompt used for determining
                                                                #   strings the user must provide in order to cmoplete and generate the prompt.
    __RulesDict: dict = {   
        __RulesRole : "",                   #   PromptType.ROLE
        __RulesInstruction : "",            #   PromptType.INSTRUCTION
        __RulesTrainingPrompt : "",         #   PromptType.TRAINING_PROMPT
        __RulesTestingPrompt : "",          #   PromptType.TESTING_PROMPT
        __RulesConfirmationPostive : "",    #   PromptType.CONFIRMATION_POSTITIVE
        __RulesConfirmationNegative : "",   #   PromptType.CONFIRMATION_NEGATIVE
        __RulesParam : ""                   #   Param symbol for prompts in the dictionary          
    }

    #
    #   Defaults (not meant to be changed)
    #
    __defaultParamSymbol: str = "_"

    #
    #   Fields
    #
    __promptDictionary: dict    # Represents the unique set of prompts contained in the prompt dictionary in the form (key: int, value: Prompt)
                                #   where the key is the id of the prompt



    #
    #   Constructors
    #
    #   Default
    def __init__(self):
        self.__promptDictionary = dict()
        self.__paramSymbol = self.__defaultParamSymbol

    
    #


    #
    #   Setters & Getters
    #
    """
    # Prompt within Prompt Dictionary
    #   Returns Prompt if found within the object's prompt dictionary
    #   Returns None otherwise
    """
    def GetPromptByID(self, id: int) -> Prompt:
        if (id in self.__promptDictionary.keys()):
            return self.__promptDictionary[id]
        else:
            SystemMessages.PromptDictionary_PromptNotFound(id)
            return None

    """
    # Prompt list containing each prompt with inputted type
    #   Returns a list of Prompts with PromptType target type if at least 1 prompt with that type exists within the dictionary
    #   Returns None otherwise
    """
    def GetPromptByType(self, targetType: PromptType) -> list:
        foundPrompts = list()
        for prompt in list(self.__promptDictionary.values()):
            if (prompt.GetPromptType() == targetType):
                foundPrompts.append(prompt)
        if (len(foundPrompts) == 0):
            SystemMessages.PromptDictionary_PromptTypeNotFound(targetType.name)
            return None
        else:
            return foundPrompts
        

    """
    #   Used for testing
    """
    def GetDict(self) -> dict:
        return self.__promptDictionary.copy()
    
    #
    #   Methods
    #

    """
    #   Given an inputted .csv file path (as a str)
    #       Attempts to open that file and load prompts into this object's dictionary
    #   Returns:
    #   True if successful in loading the prompts
    #   False otherwise
    """
    def Load(self, filePath: str, debug: bool):

        # Locals
        streamReader = None
        buffer: str = ""
        bufferTokens: list = []
        token: str = ""
        ruleFound: bool = False
        invalidType: bool = False
        state: int = -1  # Indicates while reading whether the reader is in the rules or prompts section; 0 = rules, 1 = prompts
        endOfFile: bool = False
        newPrompt: Prompt
        newPromptType: PromptType
        newPromptString: str
        counter: int = 0

        # Attempt to open the file from the path provided.
        #   Return false if file cannot be opened/isn't found
        if (not os.path.exists(filePath)):
            SystemMessages.FileNotFound(filePath)
            return False
        else:
            streamReader = open(filePath, "r")
        
        
       
        # Begin reading line by line, perform actions based on what the line encountered means
        while (not endOfFile):

            # Read the next line.
            buffer = streamReader.readline()
            
            if (debug):
                print(buffer)

            # Tokenize the buffer
            bufferTokens = buffer.split(",")


            # CASE EOF
            #   If empty, set endOfFile to true and stop looping
            if (buffer == ""):
                endOfFile = True
                if (debug):
                    print("---End of File encountered")
                
            # CASE Rules section begins
            elif (self.__RulesToken in bufferTokens):
                state = 0
                if (debug):
                    print("---Rule section encountered")

            # CASE Prompt section begins
            elif (self.__PromptsToken in bufferTokens):
                state = 1
                if (debug):
                    print("---Prompt section encountered")

            # CASE line encountered
            else:

                # If empty or of length 1 (invalid), break. Otherwise continue
                if (len(bufferTokens) <= 1):
                    continue

                # CASE reader is in rules section (state = 0)
                elif (state == 0):

                    if (debug):
                        print("---Reading rule")
                
                    # Determine which rule it is symbol as a value to its appropriate key. If it doesn't exist, continue
                    token = bufferTokens[0]
                    ruleFound = False
                    for rule in self.__RulesDict.keys():
                        if (token == rule):
                            self.__RulesDict[rule] = bufferTokens[1]
                            ruleFound = True
                            break
                    if (not ruleFound):
                        SystemMessages.PromptDictionary_InvalidRule(token, self.__RulesDict)
                
                # CASE reader is in the prompt section (state = 1)
                elif (state == 1):

                    if (debug):
                        print("---Reading prompt")
                        
                    # Verify 3 tokens exist in the buffer. If not, break. Otherwise continue
                    if (len(bufferTokens) < 3):
                        continue

                    # Determine PromptType from first token. If the prompt type doesn't exist, break. Otherwise continue
                    elif (not bufferTokens[0] in list(self.__RulesDict.values())):
                        SystemMessages.PromptDictionary_InvalidRule(bufferTokens[0], self.__RulesDict)
                        continue

                    else:
                        
                        # Get PromptType
                        invalidType = False
                        if (bufferTokens[0] == self.__RulesDict[self.__RulesRole]):
                            newPromptType = PromptType.ROLE
                        elif (bufferTokens[0] == self.__RulesDict[self.__RulesConfirmationNegative]):
                            newPromptType = PromptType.CONFIRMATION_NEGATIVE
                        elif (bufferTokens[0] == self.__RulesDict[self.__RulesConfirmationPostive]):
                            newPromptType = PromptType.CONFIRMATION_POSITIVE
                        elif (bufferTokens[0] == self.__RulesDict[self.__RulesInstruction]):
                            newPromptType = PromptType.INSTRUCTION
                        elif (bufferTokens[0] == self.__RulesDict[self.__RulesTestingPrompt]):
                            newPromptType = PromptType.TESTING_PROMPT
                        elif (bufferTokens[0] == self.__RulesDict[self.__RulesTrainingPrompt]):
                            newPromptType = PromptType.TRAINING_PROMPT
                        else:
                            # If for some reason not found (likely because of a future change). Notify and break.
                            SystemMessages.PromptDictionary_UnknownPromptType(bufferTokens[0], self.__RulesDict)
                            invalidType = True
                            continue

                        if (not invalidType):

                            # Create the Prompt. Load it into the dictionary afterwards with key equal to its id
                            try:

                                # The prompt is created from all tokens past bufferTokens[1]
                                counter = 3
                                newPromptString = bufferTokens[2]
                                while (counter < len(bufferTokens)-1):
                                    ",".join(newPromptString, bufferTokens[counter])
                                    counter+=1

                                newPrompt = Prompt(
                                    bufferTokens[2], 
                                    " ",
                                    int(bufferTokens[1]),
                                    newPromptType,
                                    self.__RulesDict[self.__RulesParam]
                                    )
                                self.__promptDictionary.update({newPrompt.GetID() : newPrompt})
                            except:
                                SystemMessages.PromptDictionary_FailToCreatePrompt(bufferTokens)
                                continue
            
        streamReader.close()                
        return True
#