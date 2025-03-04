#
#   File: Main.py
#   Description: Facilitate main execution
#

# Imports
#   Local
from AIModule import AIModel, AIWrapper, ConversationRole
from Data import LemmaDictionary, PromptDictionary, TestSystem
from DataHelpers import CleanValue, Lemma, Prompt, PromptType, WordType, Language

#   External
import random
import os
import warnings
import tiktoken
import sklearn
import sklearn.utils

# Main Execution
def main():

    # Locals
    testName: str
    testNumber: str
    instructionPrompts = list()
    debug: bool
    batchParameters = dict()
    dfs = None
    promptsUsed = []
    counter = 0
    testFileName: str
    trainingBatches: list = list()
    testBatch: list = list()
    description: str
    
    # Debug tag
    debug: bool = False # Toggle this to debug the code!

    # Directory to store test results
    testBaseDirectory = os.getcwd() + '\\TestResults'

    # AI Mode
    modelUsing: AIModel = AIModel.GPT_4_MINI # Represents the model used in main! Only 1 at a time.

    #### Critical Section, Leave Unchanged ####
    # Locals
    promptDictionary: PromptDictionary = PromptDictionary()
    lemmaDictionary: LemmaDictionary = LemmaDictionary()
    AIClient: AIWrapper = AIWrapper(modelUsing)

    # Initialize system components
    init(lemmaDictionary, promptDictionary, AIClient, debug)
    #### End Critical Section ####



    # Name of test
    testName = "Final"   # Name of tests as they are logged
    testNumber = 1      # Test number

    # Output results file name
    finalTestFileOuputName = "Final_Test_Results"

    # Name of partition to test on
    finalTestFileInputName = ""

    # List of instructions to send
    instructionPrompts = [
        promptDictionary.GetPromptByID(7),
        promptDictionary.GetPromptByID(6),
        promptDictionary.GetPromptByID(8),
        promptDictionary.GetPromptByID(9)
    ]
    batchParameters = {
        "logTraining":True,                                                     # If set to true, training results are logged as though it were a test in the lemma dictionary
        "exampleSentenceMasks":True,                                            # If set to true, keeps [MASK] tokens seen in example sentences when not all sentence words are known. Otherwise, removes said tokens
        "trainingWithQuestions":True,                                           # Represents if the model is training and whether it is training with training Statements
        "trainingPrompt":promptDictionary.GetPromptByID(10),                         # Represents the training statement for the batch without example sentences
        "trainingWithStatements":True,                                          # Represents if the model is training and whether it is training with training questions
        "testPrompt":promptDictionary.GetPromptByID(11),                     # Represents the training question for the batch without example sentences
        "exampleSentenceContextPrompt":promptDictionary.GetPromptByID(13),              # Represents the prompt that conveys the example sentence.
        "languageContextPrompt": None, #promptDictionary.GetPromptByID(12),
        "negativeReinforcementPrompt":promptDictionary.GetPromptByID(14),        # Represents the generic negative reinforcement prompt while training. If set to None, no negative reinforcement applies
        "smallMistakeCorrectionPrompt": None, #promptDictionary.GetPromptByID(15),      # Represents the negative reinforcement prompt while training when errors are off by a small amount. If set to None, small correction negative reinforcement doesn't apply
        "smallMistakeLevenshteinThreshold":3,                                   # Represents the levenshtein distance or lower of the response to the correct lemma to be considered a small mistake.
        "commonMistakeCorrectionPrompt": None, #promptDictionary.GetPromptByID(16),     # Represents the negative reinforcement prompt while training when the same incorrect lemma is guessed in training. If set to None, no negative reinforcement applies
        "commonMistakeFrequencyThreshold": 4,                                    # Represents the number of times a mistake needs to appear to be considered common
        "positiveReinforcementPrompt": None #promptDictionary.GetPromptByID(17)
    }
    description = "final test, same as gpt_4_mini_small test 7."
    

    ####################################
    #   Begin Train Test Pipeline      
    ####################################

    
    ####################################
    #   Setup
    #       Reset variables
    #       Generate training batch(es)
    #       Generate test batch   
    ####################################
    

    # Reset locals
    trainingBatches = list()
    testBatch = list()

    # Determine test file name
    testFileName = testName + '_' + str(testNumber)

    # Generate list of prompts used:
    #   Instruction prompts
    counter = 0 
    for instruction in instructionPrompts:
        counter = counter + 1
        promptsUsed.append(["instruction " + str(counter) + ": ", instruction.GetTokens()])

    #   Additional prompts
    if batchParameters["trainingPrompt"] is not None:
        promptsUsed.append(["training statement prompt: ", batchParameters["trainingPrompt"].GetTokens()])
    if batchParameters["testPrompt"] is not None:
        promptsUsed.append(["test question prompt: ", batchParameters["testPrompt"].GetTokens()])
    if batchParameters["languageContextPrompt"] is not None:
        promptsUsed.append(["language context prompt: ", batchParameters["languageContextPrompt"].GetTokens()])
    if batchParameters["exampleSentenceContextPrompt"] is not None:
        promptsUsed.append(["example sentence context prompt: ", batchParameters["exampleSentenceContextPrompt"].GetTokens()])
    if batchParameters["negativeReinforcementPrompt"] is not None:
        promptsUsed.append(["negative reinforcement prompt: ", batchParameters["negativeReinforcementPrompt"].GetTokens()])
    if batchParameters["smallMistakeCorrectionPrompt"] is not None:
        promptsUsed.append(["small mistake negative reinforcement prompt: ", batchParameters["smallMistakeCorrectionPrompt"].GetTokens()])
    if batchParameters["commonMistakeCorrectionPrompt"] is not None:
        promptsUsed.append(["common mistake negative reinforcement prompt: ", batchParameters["commonMistakeCorrectionPrompt"].GetTokens()])

    # Generate partitions
    # dfs = lemmaDictionary.PartitionDF([0.8, 0.2])

    # Generate testbatch
    # testBatch = lemmaDictionary.LemmaQuery(
    #     selectDF=dfs[2]
    #     )[0:30]
    
    # Generate training batch(es)
    #   Alternate training batches
    frequentLemmas = lemmaDictionary.LemmaQuery(
        filters=['L_appearances > 100'],
        orderBy=['L_appearances'],
        )[0:32]
    i = 0
    while i + 4 < len(frequentLemmas) and i <= 16:
        trainingBatches.append(
            frequentLemmas[i:i+4]
        )
        i = i + 4

    #   Second training batches
    infrequentLemmas = lemmaDictionary.LemmaQuery(
        filters=['L_appearances < 100'],
    )[0:50]
    j = 0
    while j + 4 < len(infrequentLemmas) and j <= 8:
        trainingBatches.append(
            infrequentLemmas[j:j+4]
        )
        j = j + 4

    
    #   Third training batches
    while i + 4 < len(frequentLemmas) and i <= 32:
        trainingBatches.append(
            frequentLemmas[i:i+4]
        )
        i = i + 4

    #   Fourth training batches
    while j + 4 < len(infrequentLemmas) and j <= 16:
        trainingBatches.append(
            infrequentLemmas[j:j+4]
        )
        j = j + 4
    

    #   Fifth training batch, include all sumerian and emesal cases
    trainingBatches.append(
        lemmaDictionary.LemmaQuery(
        filters=['L_language == "EMESAL"'],
        ) +
        lemmaDictionary.LemmaQuery(
        filters=['L_language == "SUMERIAN"'],
        )
    )


    ####################################
    #   Pipeline execution
    #       Send instructions
    #       Send training batch(es)  
    #       Store training conversation 
    #       Store training results if applicable
    #       Send test batch
    #       Store test results
    ####################################

    # Initial instructions
    for instruction in instructionPrompts:
        AIClient.SendString(instruction.ProducePrompt([]), role=ConversationRole.SYSTEM, saveToConversation=True)

    # Send training batches through
    switch = False
    batchParameters["trainingWithQuestions"] = True
    batchParameters["trainingWithStatements"] = False
    for batch in trainingBatches:
        print("############\tBegin Batch\t############")
        print("Batch count: " + str(len(batch)))

        if switch:
                batchParameters["trainingWithQuestions"] = True
                batchParameters["trainingWithStatements"] = False
                switch = False
        else:
                batchParameters["trainingWithQuestions"] = False
                batchParameters["trainingWithStatements"] = True
                switch = True

        AIClient.tt(lemmaDictionary, batch, promptDictionary, False, batchParameters)
        print("############\tBatch End\t############")

    # After training, store training results. Flush tests after.
    lemmaDictionary.StoreTestData(
        testBaseDirectory,
        testFileName + '_Training_Results',
        testName=testFileName,
        modelName=modelUsing.name,
        promptsUsed=promptsUsed
    )
    lemmaDictionary.FlushTests()

    # Save conversation to test directory
    AIClient.conversation_to_csv(testBaseDirectory + '\\' + testFileName + "_Training_Conversation")

    # Send test batch
    # AIClient.tt(lemmaDictionary, testBatch, promptDictionary, True, batchParameters)

    # # Save results to test directory
    # lemmaDictionary.StoreTestData(
    #     testBaseDirectory,
    #     testFileName + '_Results',
    #     testName=testFileName,
    #     modelName=modelUsing.name,
    #     promptsUsed=promptsUsed,
    #     description=description
    # )

    # Store test results
    testDatasetFilePathTokens = os.getcwd().split("\\")
    testDatasetFilePathTokens.pop()
    testDatasetFilePathTokens.append("Datasets")
    testDatasetFilePathTokens.append(finalTestFileInputName + ".csv")
    testDatasetFilePath = "\\".join(testDatasetFilePathTokens)
    AIClient.FullTest(testDatasetFilePath, promptDictionary.GetPromptByID(18), os.getcwd(), finalTestFileOuputName, debug=False)




################
# End Main



#
#   Initializes components used in research including the:
#       Lemma Dictionary
#       Prompt Dictionary
#       OpenAIWrapper
#   by running code on mutable objects inputted into the function for use in main().
def init(newLemmaDictionary: LemmaDictionary, newPromptDictionary: PromptDictionary, newAIWrapper: AIWrapper, debug: bool):
    
    # Locals
    promptDictionaryFilePath: str
    promptDictionaryFilePathTokens: list
    lemmaDictionaryFilePath: str
    lemmaDictionaryFilePathTokens: list

    #   Notify initialization
    print("Initializing components...")

    #   Load Prompt Dictionary
    print("\t*****Prompt Dictionary Init")
    promptDictionaryFilePathTokens = os.getcwd().split("\\")
    promptDictionaryFilePathTokens.pop()
    promptDictionaryFilePathTokens.append("Datasets")
    promptDictionaryFilePathTokens.append("Prompt Engineering Data Design Prototype.csv")
    promptDictionaryFilePath = "\\".join(promptDictionaryFilePathTokens)
    if debug:
        print("Prompt Dictionary File Path: " + promptDictionaryFilePath)
    newPromptDictionary.Load(promptDictionaryFilePath, debug)
    print("\t\tPrompt Dictionary Finished Initializing")

    #   Load Lemma Dictionary
    print("\t*****Lemma Dictionary Init")
    lemmaDictionaryFilePathTokens = os.getcwd().split("\\")
    lemmaDictionaryFilePathTokens.pop() 
    lemmaDictionaryFilePathTokens.append("Datasets")
    lemmaDictionaryFilePathTokens.append("lemmatization_train_no_ids_.csv")
    lemmaDictionaryFilePath = "\\".join(lemmaDictionaryFilePathTokens)
    if debug:
        print("Lemma Dictionary File Path: " + lemmaDictionaryFilePath)
    newLemmaDictionary.LoadDictionary(lemmaDictionaryFilePath, debug)
    levenshteinDistanceToLemmaFilePathTokens = os.getcwd().split("\\")
    levenshteinDistanceToLemmaFilePathTokens.append("Levenshtein Generator") 
    levenshteinDistanceToLemmaFilePathTokens.append("dist_to_lemmas.csv")
    levenshteinDistanceToLemmaFilePath = "\\".join(levenshteinDistanceToLemmaFilePathTokens)
    newLemmaDictionary.LoadLevenshteinDistanceToLemma(levenshteinDistanceToLemmaFilePath, debug)

    # Load dists to siblings by reading clusters file
    levenshteinDistanceToSiblingsFilePathTokens = os.getcwd().split("\\")
    levenshteinDistanceToSiblingsFilePathTokens.append("Levenshtein Generator")
    levenshteinDistanceToSiblingsFilePathTokens.append("clusters_new.csv")
    levenshteinDistanceToSiblingsFilePath = "\\".join(levenshteinDistanceToSiblingsFilePathTokens)
    newLemmaDictionary.LoadLevenshteinDistanceToSiblings(levenshteinDistanceToSiblingsFilePath, debug)

    newLemmaDictionary.StoreDictionary(os.getcwd(), "Evacun_out_new", debug) #Prompt Engineering Project
    print("\t\tLemma Dictionary Finished Initializing")


    #   Initialize OpenAIWrapper
    print("\t*****OpenAIWrapper Init")
    #newAIWrapper = AIWrapper(newAIModel)
    print("\t\tOpenAIWrapper Finished Initializing")

    #   Notify components finished initializing
    print("System finished initializing. Ready to go!")

    # Return when finished
    return
################ End Init


# Execute Main
main()
