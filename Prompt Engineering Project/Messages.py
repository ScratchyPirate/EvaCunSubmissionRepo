#
#   File: Messages.py
#   Description: Contains messages displayed to the user under different levels of severity when a notable event occurs
#       in the system.
#

#
#   Contains messages used across the system for debugging, warning, and system status communication
#
class SystemMessages:

    #   Represents the message tag seen at the beginning of each system message
    messageTag = "<Project Message>: "


    #
    #   Represents the parent method for all system print messages.
    #
    @staticmethod
    def __ParentPrint(message: str):
        print(SystemMessages.messageTag + message)

    #
    #   Represents a generic system message. 
    #       Used at debug points and other testing needs.
    # 
    @staticmethod
    def GenericMessage():
        SystemMessages.__ParentPrint("Ignore")
    ###


    #
    #   Called within any object class when attempting to load a file and that file is found to not exist or can't be opened
    #
    @staticmethod
    def FileNotFound(filePath: str):
        SystemMessages.__ParentPrint("File with path <" + filePath + "> not found.")

    #
    #   Called within any objcet class when attempting to use a directory and that directory doesn't exist or can't be opened
    #
    @staticmethod
    def DirectoryNotFound(dirPath: str):
        SystemMessages.__ParentPrint("Directory with path <" + dirPath + "> not found.")

    #
    #   Called within CV class when a sibling CV isn't found within its sibling dictionary
    #
    @staticmethod
    def CV_SiblingNotFound(source: str, targetSibling: str):
        SystemMessages.__ParentPrint("Failed to find sibling <" + targetSibling + "> in source dictionary of <" + source + ">.")
    
    #
    #   Called within CV class when inserting an example sentence and that sentence doesn't contain the CV
    #
    @staticmethod
    def CV_InvalidExampleSentence(source: str, sentence: str):
        SystemMessages.__ParentPrint("Invalid CV example sentence. <" + sentence + "> doesn't contain <" + source + ">")

    #
    #   Called within CV class when popping an example sentence and the list of example sentences is empty
    #
    @staticmethod
    def CV_PopFromEmptySentenceList(source: str):
        SystemMessages.__ParentPrint("CV. <" + source + "> example sentence list empty.")

    #
    #   Called within Lemma class when a child CV isn't found within its CV dictionary
    #
    @staticmethod
    def L_CVNotFound(source: str, cvName: str):
        SystemMessages.__ParentPrint("Failed to find child CV <" + cvName + "> in source lemma <" + source + ">.")

    #
    #   Called within LemmaDictionary class when a child Lemma isn't found within its Lemma dictionary
    #
    @staticmethod
    def LD_LNotFound(lName: str):
        SystemMessages.__ParentPrint("Failed to find child Lemma <" + lName + "> in LemmaDictionary.")


    #
    #   Called within LemmaDictionary class when trying to load lemma dataset and the file inputted is in an invalid format
    #
    @staticmethod
    def LD_InvalidFileFormat(filePath: str):
        SystemMessages.__ParentPrint("Invalid file format read in LemmaDictionary. File <" + filePath + "> unable to be read or empty")

    #
    #   Called within LemmaDictionary class when trying to query the dictionary and the dictionary hasn't been stored as a .csv file
    #
    @staticmethod
    def LD_InvalidQueryStoreNotRan():
        SystemMessages.__ParentPrint("Query cannot be ran until dictionary is stored in a .csv file.")


    #
    #   Called within TestDictionary class when trying to read test data and the file is in an invalid format
    #
    @staticmethod
    def TS_InvalidFileFormat(filePath: str):
        SystemMessages.__ParentPrint("Invalid file format read in TestSystem. File <" + filePath + "> unable to be read or empty")
       
    #
    #   Called within TestDictionary class when trying to read test data and the file is in an invalid format
    #
    @staticmethod
    def TS_NoPackets():
        SystemMessages.__ParentPrint("No packets remain to be read from TestSystem.")
       
    #
    #   Called within TestDictionary class when trying to write test data and the file path cannot be used (likely due to the directory missing or the file being openned)
    #
    @staticmethod
    def TS_InvalidFileFormat(filePath: str):
        SystemMessages.__ParentPrint("Invalid file format read in TestSystem. File <" + filePath + "> unable to be read or empty")
       

    #
    #   Called within Prompt class when not enough parameters are given to a prompt when producing its prompt string
    #
    @staticmethod
    def Prompt_InsufficientTokensProvided(promptTokens: list, promptParams: int, promptFillerSymbol: str, paramsProvided: list):
        SystemMessages.__ParentPrint("Insufficient tokens provided to prompt " + promptFillerSymbol.join(promptTokens))
        SystemMessages.__ParentPrint("  Required: " + str(promptParams))
        SystemMessages.__ParentPrint("  Provided: " + str(len(paramsProvided)) + "; " + "|".join(paramsProvided))

    #
    #   Called within PromptDictionary class when attempting to retrieve a prompt by id that doesn't exist
    #   
    @staticmethod
    def PromptDictionary_PromptNotFound(promptID: int):
        SystemMessages.__ParentPrint("Prompt with ID <" + str(promptID) + "> not found within PromptDictionary.")

    #
    #   Called within PromptDictionary class when attempting to retrieve a list of prompts by PromptType and no prompts of that type are found
    #
    @staticmethod
    def PromptDictionary_PromptTypeNotFound(promptTypeAsStr: str):
        SystemMessages.__ParentPrint("Prompts with type <" + promptTypeAsStr + "> not found within PromptDictionary.")

    #
    #   Called within PromptDictionary class when attempting to load prompts and an unknown rule token was found in the rules or prompt section
    #
    @staticmethod
    def PromptDictionary_InvalidRule(ruleToken: str, rulesDict: dict):
        SystemMessages.__ParentPrint("Invalid rule token <" + ruleToken + "> found and could not be associated with existing rules.")
        SystemMessages.__ParentPrint("Existing rules: " + str(rulesDict))

    #
    #   Called within PromptDictionary class when attempting to load prompts and an unknown prompt type is encountered
    #
    @staticmethod
    def PromptDictionary_UnknownPromptType(typeToken: str, rulesDict: dict):
        SystemMessages.__ParentPrint("Unkown prompt token <" + typeToken + "> that could not be associated with a prompt type.")
        SystemMessages.__ParentPrint("Existing rules: " + str(rulesDict))

    #
    #   Called within PromptDictionary class when attempting to create a prompt from loaded file information and that process fails
    #
    @staticmethod
    def PromptDictionary_FailToCreatePrompt(bufferTokens: str):
        SystemMessages.__ParentPrint("Failed to create prompt with tokens: " + "|".join(bufferTokens))

    #
    #   Called within PromptDictionary class when attempting to insert a prompt and a prompt with the same key exists already
    #
    @staticmethod
    def PromptDictionary_FailToInsertPrompt(bufferTokens: str):
        SystemMessages.__ParentPrint("Failed to insert prompt with tokens: " + "|".join(bufferTokens) + "(ID already exists within dictionary)")

    #
    #   Called within AIWrapper class when attempting to send a message to a model and the model isn't supported.
    #
    @staticmethod
    def AIWrapper_ModelNotSupported(modelName: str):
        SystemMessages.__ParentPrint("Could not send message. Model <" + modelName + "> not supported")

    #
    #   Called within AIWrapper class when recieving a message and the message was found to be empty.
    #
    @staticmethod
    def AIWrapper_BadResponse(requestString: str):
        SystemMessages.__ParentPrint("No response recieved on most recent request <" + requestString + ">")

    #
    #   Represents the AIWrapper when converting a string to token count and the tokenizer is not supported
    #
    @staticmethod
    def AIWrapper_TokenizerNotSupported(model: str):
        SystemMessages.__ParentPrint("Model doesn't support a tokenizer <" + model + ">")

    #
    #   Called from the AIWrapper when the AIWrapper FullTest() method is called and the TestSystem cannot be initialized.
    #
    @staticmethod
    def AIWrapper_TestSystemFailedToLoad():
        SystemMessages.__ParentPrint("AIWrapper failed to initialize the test system.")

    #
    #   Called from the AIWrapper when the AIWrapper tt() method is called and no batch parameters are provided
    #
    @staticmethod
    def AIWrapper_NoBatchParameters():
        SystemMessages.__ParentPrint("No batch parameters provided. Failed to run batch.")

###
# End SystemMessages

#
#   Contains messages used across the system for error/exception communication
#
class ErrorMessages:

    messageTag = "<Project Error>: "

    #
    #   Represents the parent method for all system print messages.
    #
    @staticmethod
    def __ParentPrint(message: str):
        print(ErrorMessages.messageTag + message)

    #
    #   Represents a generic error message. 
    #       Used at error points and when a specific error message isn't defined.
    # 
    @staticmethod
    def GenericError():
        ErrorMessages.__ParentPrint(" Exception Occured.")

    #
    #   Represents the AIWrapper when initializing and the model used is not supported
    #
    @staticmethod
    def AIWrapper_ModelNotSupported(model: str):
        ErrorMessages.__ParentPrint(" AI model <" + model + "> unsupported by AI wrapper.")


    #
    #   Represents the AIWrapper when sending a string and an exception occurs
    #
    @staticmethod
    def AIWrapper_SendStringException(model: str):
        ErrorMessages.__ParentPrint("Exception occured while sending string to model <" + model + ">.")


###
# End ErrorMessages


