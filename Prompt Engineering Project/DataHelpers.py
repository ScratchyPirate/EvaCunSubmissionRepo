#
#   File: DataHelpers.py
#   Description: Contain System classes and methods that represent
#       smaller containers and functions used in Data.py for data
#       retrieval and storage purposes. Includes testing features
#       for relevant objects.
#

# Imports
#   Local
from Messages import SystemMessages

#   External
from enum import Enum
from typing import Final

# Represents types of words
class WordType(Enum):
    UNKNOWN = 0,
    NOUN = 1,
    ADJECTIVE = 2,
    VERB = 3,
    PRONOUN = 4
# Represents possible languages of lemmas
class Language(Enum):
    UNKNOWN = 0,
    AKKADIAN = 1,
    SUMERIAN = 2,
    EMESAL = 3
# Represents types of prompts stored for use in prompt engineering
class PromptType(Enum):
    ROLE = 1
    INSTRUCTION = 2
    TRAINING_PROMPT = 3
    CONFIRMATION_POSITIVE = 4
    CONFIRMATION_NEGATIVE = 5
    TESTING_PROMPT = 6
# Generic Event Class
class Event(object): 
	def __init__(self): 
		self.__eventHandlers = [] 
	def __iadd__(self, Ehandler): 
		self.__eventHandlers.append(Ehandler) 
		return self
	def __isub__(self, Ehandler): 
		self.__eventHandlers.remove(Ehandler) 
		return self
	def __call__(self, *args, **keywargs):  
            for eventHandler in self.__eventHandlers:
                eventHandler(*args, **keywargs) 
            return


# Constants associated with CV
CVExampleSentenceLimit: Final[int] = 15 # The maximum number of example sentences a CV can hold. 
CVExampleSentenceMaskToken: Final[str] = "[MASK]" # To place in sentences when the full sentence is available only in fragments and blanks need to be filled in.
#
#   Represents a Clean Value (CV) word
#
class CleanValue:

    #
    #   Fields
    #   
    # For literal data:
    __value: str    # Represents the CV String
    __exampleSentences: list # Represents a list of sentences the CV is used in

    # For statistical data:
    __levenshteinDistanceToParent: float   # Represents the distance between the CV and its lemma (L) by way of transformations and changes required to go from the CV to L (and vice versa).
    __levenshteinDistanceToSiblings: dict    # Represents the distance between the CV and sibling CVs.
    __relativeFrequency: float  # Represents the frequency in appearance of th CV in relation to other CVs seen with the lemma
    __occurences: int   # The number of times the CV appears in its dataset

    # For testing:
    __testedFlag: bool  # Represents whether the clean value was used in testing. Defaults to false.
    __testCorrectGuessCount: int # Represents the number of times this CV was correctly guessed by an AI model.
    __testIncorrectGuessCount: int # Represents the number of times this CV was incorrectly guessed by an AI model.
    __testIncorrectGuessWords: list # Represents the words that were incorrect guesses of the CV's lemma

    #   Events
    __onOccurencesChanged: Event # Represents when the number of occurences changes in the cv.
    __onTestCorrect: Event # Represents when the CV is tested and the result is correct.
    __onTestIncorrect: Event # Represents when the CV is tested and the result is not correct.

    #
    #   Constructors
    #
    #   Default Constructor
    def __init__(self, newCV: str):
        self.__value = newCV
        self.__occurences = 1
        self.__levenshteinDistanceToParent = 0
        self.__levenshteinDistanceToSiblings = {}
        self.__relativeFrequency = 0
        self.__exampleSentences = []
        self.__testCorrectGuessCount = 0
        self.__testIncorrectGuessCount = 0
        self.__testedFlag = False
        self.__testIncorrectGuessWords = []

        self.__onOccurencesChanged = Event()
        self.__onTestCorrect = Event()
        self.__onTestIncorrect = Event()
        return
    #

    #
    #   Setters & Getters
    #

    # Value
    def GetValue(self) -> str:
        return self.__value 
    
    # Occurences
    def GetOccurences(self) -> int:
        return self.__occurences
    def SetOccurences(self, newOccurences: float):
        self.__occurences = newOccurences

    # Levenshtein Distance to Parent
    def GetLevenshteinDistanceToParent(self) -> float:
        return self.__levenshteinDistanceToParent
    def SetLevenshteinDistanceToParent(self, newDistance: float):
        self.__levenshteinDistanceToParent = newDistance

    # Levenshtein Distance to Sibling
    def GetLevenshteinDistanceToSiblingDictionary(self) -> dict:
        return self.__levenshteinDistanceToSiblings.copy()
    """ 
    #   Returns:
    #       Distance (positive) if key is found in dictionary
    #       None if key isn't found in dictionary
    """
    def GetLevenshteinDistanceToSibling(self, key: str) -> float:
        if (key in self.__levenshteinDistanceToSiblings.keys()):
            return self.__levenshteinDistanceToSiblings[key]
        else:
            SystemMessages.CV_SiblingNotFound(self.__value, key)
            return None
    def SetLevenshteinDistanceToSibling(self, key: str, distance: float):
        if (not key in self.__levenshteinDistanceToSiblings.keys()):
            self.InsertSibling(key)
        self.__levenshteinDistanceToSiblings[key] = distance

    # Frequency
    def GetRelativeFrequency(self) -> float:
        return self.__relativeFrequency
    def SetRelativeFrequency(self, newFrequency: float):
        self.__relativeFrequency=newFrequency

    # Sentences
    def GetExampleSentences(self, includeMaskTokens: bool = True) -> list:

        # Get a temporary copy of example
        returnList = self.__exampleSentences.copy()

        # If mask tokens are excluded, then remove them from the return list.
        if (not includeMaskTokens):
            for i in range(0, len(returnList)):
                returnList[i] = returnList[i].replace(CVExampleSentenceMaskToken + ' ', '')

        return returnList

    # Test flag
    def Tested(self) -> bool:
        return self.__testedFlag
    
    # Correct Guesses
    def TestCorrectGuessCount(self) -> int:
        return self.__testCorrectGuessCount

    # Incorrect Guesses
    def TestIncorrectGuessCount(self) -> int:
        return self.__testIncorrectGuessCount
    
    # Incorrect Guess Words
    def TestIncorrectGuessWords(self) -> list:
        return self.__testIncorrectGuessWords.copy()

    #
    #   Methods
    #

    """
    #   Purpose of providing an incrementer to CV
    """
    def IncrementOccurences(self):
        self.__occurences+=1
    
    """
    #   Inserts a new sibling CV into this CV's sibling dictionary
    #       -1 implies uninitialized.
    """
    def InsertSibling(self, newCV: str):
        self.__levenshteinDistanceToSiblings[newCV] = -1

    """
    #   Allowed list methods for example sentence list
    """
    #   Inserts the sentence only if it contains the CV
    def PushExampleSentence(self, newSentence: str):
        if self.__value in newSentence:
            # If the limit has been reached for example sentences, don't add any more
            if CVExampleSentenceLimit == len(self.__exampleSentences):
                return
            else:
                self.__exampleSentences.append(newSentence)
        else:
            SystemMessages.CV_InvalidExampleSentence(self.__value, newSentence)
    #   Pops only if len > 0
    def PopExampleSentence(self) -> str:
        if len(self.__exampleSentences) <= 0:
            SystemMessages.CV_PopFromEmptySentenceList(self.__value)
        else:
            return self.__exampleSentences.pop()
    #   Gets the number of example sentences
    def GetExampleSentenceCount(self) -> int:
        return len(self.__exampleSentences)




    #
    #   Event handler subscriptions
    #
    # Event on occurences changed
    def SubscribeOccurencesChanged(self, eventHandler):
        self.__onOccurencesChanged += eventHandler
    # Event on test correct
    def SubscribeTestCorrect(self, eventHandler):
        self.__onTestCorrect += eventHandler
    # Event on test incorrect
    def SubscribeTestIncorrect(self, eventHandler):
        self.__onTestIncorrect += eventHandler

    #
    #   Used in testing
    #
    """
    #   Log a new test entry for the CV, specifying whether the guess was correct or not.
    #       if not, specify the incorrectly guessed word for the CV's lemma
    """
    def LogTest(self, correct: bool, incorrectGuessedWord: str = None):
        # If this is the first test, mark that the CV has been tested.
        if (not self.__testedFlag):
            self.__testedFlag = True
        # If correct, increment the number of correct guesses
        if (correct):
            self.__testCorrectGuessCount+=1
            self.__onTestCorrect()
        # Otherwise, increment the number of incorrect guesses. If a word is provided, append it to the end of incorrectly guessed words.
        else:
            self.__testIncorrectGuessCount+=1
            if (incorrectGuessedWord != None):
                self.__testIncorrectGuessWords.append(incorrectGuessedWord)
            self.__onTestIncorrect()

    """
        Used for reseting testing variables
    """
    def ResetTests(self):
        if (self.__testedFlag):
            self.__testedFlag = False
            self.__testCorrectGuessCount = 0
            self.__testIncorrectGuessCount = 0
            self.__testIncorrectGuessWords = []


#
#   Represents a Lemma (L) word
#
class Lemma:

    #
    #   Fields
    #   
    # For literal data:
    __value: str    # Represents the lemma String
    __language: Language    # Represents the language of the lemma
    __wordType: WordType    # Represents the lemma's word type
    __cvs: dict     # The dictionary of CVs associated with the lemma

    # For statistical data:
    __occurences: int   # The number of times the lemma appears in its dataset
    __relativeFrequency: float  # Represents the frequency of the lemma in comparison with all other lemmas

    # For testing:
    __testedFlag: bool # Represents whether the bool was used in testing. Default to false
    __testCorrectGuessCount: int # Stores how many correct guesses have occurred across this lemma's cvs
    __testIncorrectGuessCount: int # Stores how many incorrect guesses have occured across this lemma's cvs

    #   Events
    __onOccurencesChanged: Event # Represents when the number of occurences changes in the cv.
    __onTestCorrect: Event # Represents when the CV is tested and the result is correct.
    __onTestIncorrect: Event # Represents when the CV is tested and the result is not correct.

    #
    #   Constructors
    #
    #   Default Constructor
    def __init__(self, newLemma: str):
        self.__value = newLemma
        self.__occurences = 1
        self.__cvs = {}
        self.__relativeFrequency = 0
        self.__language = Language.UNKNOWN
        self.__wordType = WordType.UNKNOWN
        self.__testedFlag = False
        self.__testCorrectGuessCount = 0 
        self.__testIncorrectGuessCount = 0 

        self.__onOccurencesChanged = Event()
        self.__onTestCorrect = Event()
        self.__onTestIncorrect = Event()


    #
    #   Setters & Getters
    #

    # Value
    def GetValue(self) -> str:
        return self.__value

    # Occurences
    def GetOccurences(self) -> int:
        return self.__occurences

    # CV Dictionary
    def GetCVDictionary(self) -> dict:
        return self.__cvs.copy() 
    """
    #    Returns:
    #       CV if key is found in dictionary
    #       None (null) if key isn't found in dictionary
    """
    def GetCV(self, key: str) -> CleanValue:
        if (key in self.__cvs.keys()):
            return self.__cvs[key]
        else:
            return None

    # Frequency
    def GetRelativeFrequency(self) -> float:
        return self.__relativeFrequency
    def SetRelativeFrequency(self, newFrequency: float):
        self.__relativeFrequency = newFrequency

    # Language
    def GetLanguage(self) -> Language:
        return self.__language
    def SetLanguage(self, newLanguage: Language):
        self.__language = newLanguage

    # Word Type
    def GetWordType(self) -> WordType:
        return self.__wordType
    def SetWordType(self, newWordType: WordType):
        self.__wordType = newWordType

    # Tested Flag
    def Tested(self) -> bool:
        return self.__testedFlag

    # Correct Guesses Count
    def TestCorrectGuessCount(self) -> int:
        return self.__testCorrectGuessCount
    
    # Total Guess Count
    def TestTotalGuessCount(self) -> int:
        return self.__testCorrectGuessCount + self.__testIncorrectGuessCount

    # Incorrect Guess Count
    def TestIncorrectGuessCount(self) -> int:
        return self.__testIncorrectGuessCount
    
    #
    #   Methods
    #

    """ 
    #   Given a string representing a CV, if that CV doesn't exist within the lemma's
    #       CV dictionary, then a new CV object is created with that string and inserted
    #       into the dictionary. Otherwise, the number of occurences of an existing CV
    #       are incremented instead.
    """     
    def InsertCV(self, key: str):
        if (key in self.__cvs.keys()):
            self.__cvs[key].IncrementOccurences()
        else:
            # Create a new CV
            newCV = CleanValue(key)

            # Subscribe to CV events
            newCV.SubscribeTestCorrect(self.__IncrementCorrectGuesses)
            newCV.SubscribeTestIncorrect(self.__IncrementIncorrectGuesses)

            # Add new CV into the dictionary
            self.__cvs[key] = newCV

    """
    #   Given a key representing a CV's key in the lemma and a new CV value,
    #       this function inserts into the sibling of the CV denoted by the key
    #       a new key denoted by newCV. If the CV denoted by the key isn't found
    #       a system message is printed.
    """
    def InsertCVSibling(self, key: str, newCV: str):
        if (key in self.__cvs.keys()):
            self.__cvs[key].InsertSibling(newCV)
        else:
            SystemMessages.L_CVNotFound(self.__value, key)            

    """
    #   Given a key representing a CV's key in the lemma and a new CV value,
    #       this function inserts into the sibling of the CV denoted by the key
    #       a new key denoted by newCV with a value of distance. If the CV denoted 
    #       by the key isn't found, a system message is printed.
    """
    def InsertCVLevenshteinDistanceToSibling(self, key: str, newCV: str, distance: float):
        if (key in self.__cvs.keys()):
            self.__cvs[key].SetLevenshteinDistanceToSibling(newCV, distance)
        else:
            SystemMessages.L_CVNotFound(self.__value, key)

    """
    #   Given a key representing a CV's key in the lemma and an example sentence
    #       with that CV in it.
    #   This function stores the example sentence in the CV's example
    #       sentence list if it is found within this lemma's CV dictionary.
    """
    def InsertCVExampleSentence(self, key: str, sentence: str):
        if (key in self.__cvs.keys()):
            self.__cvs[key].PushExampleSentence(sentence)
        else:
            SystemMessages.L_CVNotFound(self.__value, key)  

    """
    #   Purpose of providing an incrementer to Lemma
    """
    def IncrementOccurences(self):
        self.__occurences+=1
        self.__onOccurencesChanged()

    """
    #   Purpose of updating the relative frequencies of CV in the lemma.
    #   Done as a function manually executed (rather than each time a CV is inserted)
    #   for performance concerns. More efficient to run it once or twice as needed.
    """
    def UpdateCVFrequencies(self):
        tempCV: CleanValue
        for key in self.__cvs.keys():
            tempCV = self.__cvs[key]
            tempCV.SetRelativeFrequency(tempCV.GetOccurences()/self.__occurences)
            self.__cvs[key] = tempCV

    """
    #   Gets each CV in the dictionary and returns them as a list in order of frequency descending
    """ 
    def GetCVsByFrequency(self) -> list:
        return Lemma.__CVFrequencyQuickSort(list(self.__cvs.values()))
    
    """
    #   Compiles together all example sentences of the CVs the lemma contains in the form of (exampleSentence: str, cvValue: str)
    #       and returns the list
    """
    def GetExampleSentences(self) -> list:
        rl = []
        tuple = ()
        for cv in self.__cvs.values():
            for es in cv.GetExampleSentences():
                tuple = (es, cv.GetValue())
                rl.append(tuple)
        return rl

    
    #
    #   Event handler subscriptions
    #
    # Event on occurences changed
    def SubscribeOccurencesChanged(self, eventHandler):
        self.__onOccurencesChanged += eventHandler
    # Event on test correct
    def SubscribeTestCorrect(self, eventHandler):
        self.__onTestCorrect += eventHandler
    # Event on test incorrect
    def SubscribeTestIncorrect(self, eventHandler):
        self.__onTestIncorrect += eventHandler

    #
    #   Used in testing   
    #
    """
    # 
    #   Used to log a test case for a CV found within this lemma.
    #   Enter the CV's name, if the guess was correct, and optionally what the incorrectly guessed word was.
    #   The test data is then stored within the lemma's CV if found within its CV dictionary.
    #
    """
    def LogTest(self, cv: str, correct: bool, incorrectGuessedWord: str = None):
        # Mark that this object instance has been tested
        if (not self.__testedFlag):
            self.__testedFlag = True
            
        # Insert the test into the lemma's CV dictionary in its corresponding CV if found within the dictionary.
        if (self.__cvs.__contains__(cv)):
            self.__cvs[cv].LogTest(correct, incorrectGuessedWord)
        else:
            SystemMessages.L_CVNotFound(self.__value, cv)

    """
        This method resets all tests in the Lemma and its CVs to their default state:
        For self
            Tested -> False
        For CVs
            Tested -> False
            CorrectGuessesCount -> 0
            IncorrectGuessCount -> 0
            IncorrectGuessWords -> []
    """
    def ResetTests(self):
        if (self.__testedFlag):
            self.__testedFlag = False
            self.__testCorrectGuessCount = 0
            self.__testIncorrectGuessCount = 0
            for cv in self.__cvs.values():
                cv.ResetTests()
        
    
    """
        Used to increment correct/incorrect guesses in this object.
    """
    def __IncrementCorrectGuesses(self):
        self.__testCorrectGuessCount+=1
        self.__onTestCorrect()
    def __IncrementIncorrectGuesses(self):
        self.__testIncorrectGuessCount+=1
        self.__onTestIncorrect()


    """
    #   Private  
    #   Sorting algorithm that sorts CVs by frequency in descending order
    """
    def __CVFrequencyQuickSort(cvList: list) -> list:
        Lemma.__CVFrequencyQuickSortHelper(cvList, 0, len(cvList)-1)
        return cvList
    def __CVFrequencyQuickSortHelper(cvList: list, low: int, high: int) -> list:
        if low < high:
            pivot: int = Lemma.__partition(cvList, low, high)
            Lemma.__CVFrequencyQuickSortHelper(cvList, low, pivot-1)
            Lemma.__CVFrequencyQuickSortHelper(cvList, pivot+1, high)
    def __partition(cvList: list,  low: int, high: int) -> int:
        pivot: int = cvList[high]
        i: int = low -1
        j: int
        tempCV: CleanValue
        for j in range(low , high):
            if cvList[j].GetOccurences() >= pivot.GetOccurences():
                i+=1
                tempCV = cvList[i]
                cvList[i] = cvList[j]
                cvList[j] = tempCV
        tempCV = cvList[i+1]
        cvList[i+1] = cvList[high]
        cvList[high] = tempCV
        return i+1
    

#
#   Represents a Prompt (P) to be fed into an OpenAI model
#
class Prompt:

    #
    #   Defaults   (Not to be changed)
    #
    __defaultFillerSymbol: Final[str] = " "
    __defaultID: int = -1
    __defaultParams: int = 0
    __defaultParamSymbol: str = "_"
    __defaultPromptType: PromptType = PromptType.ROLE

    #  
    #   Fields
    #
    __tokens: list  # Represents the tokens that make up the prompt
    __fillerSymbol: str # Represents the filler symbol(s) that are filled in when the prompt is joined into a single string
                        # For example, a prompt with tokens ["Hello","World","My","Name","Is","John"] and filler symbol "_" 
                        # would produce a prompt string as "Hello_World_My_Name_Is_John"
    __id: int       # Represents the id number representing the prompt (intended to be unique)
    __params: int   # Represents the number of parameter tokens found within __tokens (Tokens are to be filled in by the user
                    #  when the string representing the prompt is created)
    __paramSymbol: str  # Represents the token that represents a param in the prompt's list of tokens.
    __promptType: PromptType    # Represents the type of prompt this object represents.


    #
    #   Constructors
    #
    # Default Constructor
    def __init__(self):
        self.__tokens = []
        self.__fillerSymbol = self.__defaultFillerSymbol
        self.__id = self.__defaultID
        self.__params = self.__defaultParams
        self.__paramSymbol = self.__defaultParamSymbol
        self.__promptType = self.__defaultPromptType


    # 
    #   Initializes the prompt based on the given inputs using the InitializePrompt() method
    #
    def __init__(self, newPrompt: str, delimiter: str, newID: int, newPromptType: PromptType, newParamSymbol: str):
        
        self.InitializePrompt(newPrompt, delimiter, newID, newPromptType, newParamSymbol)
        return
    #



    #
    #   Setters & Getters
    #

    #   Tokens
    def GetTokens(self) -> list:
        return self.__tokens
    
    #   Filler Symbol
    def GetFillerSymbol(self) -> str:
        return self.__fillerSymbol
    def SetFillerSymbol(self, newFillerSymbol: str):
        self.__fillerSymbol = newFillerSymbol

    #   ID
    def GetID(self) -> int:
        return self.__id
    def SetID(self, newID: int):
        self.__id = newID

    #   Prompt Type
    def GetPromptType(self) -> PromptType:
        return self.__promptType
    def SetPromptType(self, newPromptType: PromptType):
        self.__promptType = newPromptType

    #   Params
    def GetParams(self) -> int:
        return self.__params
    
    #   Param Symbol
    def GetParamSymbol(self) -> str:
        return self.__paramSymbol

    #
    #   Methods
    #

    """ 
    #   Given:
    #       newPrompt String representing the prompt
    #       newID Int representing the id of the prompt
    #       newParamSymbol String representing the param symbol
    #       newPromptType PromptType representing the prompt's type
    #       delimiter String representing the symbol used for tokenizing the prompt
    #   This method:
    #       Initializes this object instance with the inputted id and prompt type as well as
    #       parses the prompt string to determine the prompt's tokens and number of 
    #       params based on the inputted param symbol(s). Delimiter represents the symbols used to
    #       split the prompt string (and us assigned to fillerSymbol during construction).
    """
    def InitializePrompt(self, newPrompt: str, delimiter: str, newID: int, newPromptType: PromptType, newParamSymbol: str):

        # Initialize each field
        self.__tokens = []
        self.__fillerSymbol = delimiter
        self.__id = newID   # Assign ID
        self.__params = self.__defaultParams
        self.__paramSymbol = newParamSymbol
        self.__promptType = newPromptType   # Assign Prompt Type

        # Locals
        tempPromptString = newPrompt
        tempPromptTokens = []
        tempPromptTokens2 = []
        tempPromptTokens3 = []
        tempParamSymbol = newParamSymbol
        paramsFound: int = 0
        i: int = 0

        # If newPrompt is empty, halt and return (no need to proceed)
        #   Otherwise, look to tokenizing the prompt
        if (len(newPrompt) == 0):
            __params = 0
            return

        # Determine params and their location
        #   If param is within the prompt, delimit the prompt by param and store in temp prompt tokens
        #   Otherwise, make temp prompt tokens contain the prompt string
        if (tempParamSymbol in newPrompt):
            tempPromptTokens = tempPromptString.split(tempParamSymbol)

            #
            # For each string delimited by paramSymbol (or just the prompt itself) in tempPromptTokens
            #   tokensize it and store in tempPromptTokens2. If there are more than one string (param was)
            #   found, then insert the param symbol and repeat.
            #  
            i = 0 
            paramsFound: int = len(tempPromptTokens)-1
            for string in tempPromptTokens:

                # Append string to tempPromptTokens2 after delimiting into string tokens
                tempPromptTokens3 = string.split(delimiter)
                tempPromptTokens2.extend(tempPromptTokens3)

                # If there are remaining prompt strings, add a paramSymbol and proceed
                if (i < len(tempPromptTokens)-1):
                    tempPromptTokens2.append(tempParamSymbol)
                
                # Increment i
                i+=1


        else:
            tempPromptTokens2 = tempPromptString.split(delimiter)
   
            
        #   Set the prompt tokens and param count
        self.__params = paramsFound
        self.__tokens = tempPromptTokens2

        #   Return when finished
        return
    #

    """
    #   Given a list of parameter inputs (assumed to be strings)
    #   This method joins together all its tokens while applying 
    #       the inputted parameters from lowest index to highest
    #       index into the prompt from the order of the first
    #       to last parameter symbol. Produces the prompt string 
    #       when finished.
    #       *If not enough parameters are provided, the method 
    #       will produce None instead
    #       *If more parameters are provided than symbols seen
    #       in the prompt tokens, only the lowest index inputs
    #       will be applied to the prompt (in the order they are
    #       seen from lowest to highest index in normal behavior)
    """
    def ProducePrompt(self, parameterInputs: list = []) -> str:

        # If insufficient paramaters are provided. Notify and return None
        if (len(parameterInputs) < self.__params):
            SystemMessages.Prompt_InsufficientTokensProvided(self.__tokens, self.__params, self.__fillerSymbol, parameterInputs)
            return None
        
        # For each param, insert the parameterInputs into a copy of the list. Then join together with the 
        #   filler symbol.
        instanceTokens: list = self.__tokens.copy()
        promptInstance: str = ""
        j: int = 0
        for i in range(len(instanceTokens)):
            if (instanceTokens[i] == self.__paramSymbol):
                instanceTokens[i] = parameterInputs[j]
                j+=1
        promptInstance = self.__fillerSymbol.join(instanceTokens)

        # Return joined prompt when finished
        return promptInstance 
    #


		