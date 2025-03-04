
#
#   File: AIModule.py
#   Description: Contains AI wrapper classes and methods for EvaCun usage
#

# Imports
#   Local 
from Messages import SystemMessages, ErrorMessages
from Data import PromptDictionary, PromptType, LemmaDictionary, TestSystem
from DataHelpers import Prompt, Lemma, CleanValue

#   External
import csv
import sys 
import events
import os
from enum import Enum
from time import sleep
from typing import Final
import random
import sklearn
import tiktoken
import transformers
import sklearn.utils
import Levenshtein

#   AI providers
from openai import OpenAI
from anthropic import Anthropic


#
#   Defines roles of a message used in a model.
#        "system": Allows you to specify the way the model answers questions. Classic example: “You are a helpful assistant.”
#        "user": Equivalent to the queries made by the user.
#        "assistant": AI response
#
class ConversationRole(Enum):
    SYSTEM = 0,
    USER = 1,
    ASSISTANT = 2

#
#   Represents AI models useable by the AIWrapper
#
class AIModel(Enum):
    GPT_4 = 0,
    GPT_4_MINI = 1,
    O1_MINI = 2,
    DEEP_SEEK_CHAT = 3,
    CLAUDE_3_5_SONNET = 4

#
#   Represents AI API providers used in the project.
#
class AIProvider(Enum):
    OPEN_AI = 0,
    DEEP_SEEK = 1,
    ANTHROPIC = 2

#
#   The class with the purpose of facilitating communication between an openAI model
#       for this project's purposes.
#
class AIWrapper:

    #
    #   Constants
    #

    #   For OpenAI
    __projectID: Final[str] = ""   #   Project id for OpenAI
    __organizationID: Final[str] = ""   # Organization id for OpenAI
    #       API key for OpenAI
    __openAIAPIKey: Final[str] = ""
    
    #   For DeepSeek
    #       API key for DeepSeek
    __deepSeekAPIKey: Final[str] = ""
    __deepSeekBaseURL: Final[str] = "https://api.deepseek.com"

    #   For Anthropic
    #       API key for Anthropic
    __anthropicAPIKey: Final[str] = ""

    #   Dictionary of supported models with the KEY being their english name and VALUE being the encoding recognized by the API
    __supportedModels: Final[dict] = {
        AIModel.GPT_4_MINI: "gpt-4o-mini",
        AIModel.GPT_4 :"gpt-4",
        AIModel.O1_MINI :"o1-mini",
        AIModel.DEEP_SEEK_CHAT:"deepseek-chat",
        AIModel.CLAUDE_3_5_SONNET:"claude-3-5-sonnet-20241022"
    }

    #   Contains the string tokens needed to send messages to a model under different roles. Those
    #       roles are defined in the Conversation_Roles enum.
    __supportedRoles: Final[dict] = {
        ConversationRole.SYSTEM : "system",
        ConversationRole.USER : "user",
        ConversationRole.ASSISTANT : "assistant"
    }

    #
    #   Fields
    #

    # Holds the OpenAI object used for communication between the OpenAI API if using the OpenAI API
    __openAIClient: OpenAI
    __anthropicClient: Anthropic

    # Represents the AI model the AIWrapper is using.
    __model: AIModel
    # Represents the company providing the model's service
    __modelProvider: AIProvider
    # Represents the tokenizer used for the model
    __tokenizer: None   


    # Represents the current conversation held with a model. Each input and response is held in the format
    #   {"role": [ROLE TOKEN], "content": [CONTENT TOKENS]}
    #   Where:
    #       ROLE TOKEN represents the role seen in that input/output. Can be one of these three:
    #               -"system" (what we give to it for context/establish role)
    #               -"user" (what we give to it)
    #               -"assistant" (AI response))
    #       CONTENT TOKENS represents the strings given or outputted from that conversation part
    __currentConversation: list
    # Holds roughly the number of tokens currently in the conversation
    __conversationTokens: int
    # Holds system prompt for specifically the anthropic client
    __anthropicSystemConversation: str

    #
    # Constructors
    #
    # Default Constructor
    def __init__(self, wrapperModel: AIModel):

        # If the model isn't supported, throw a new error message saying so.
        if not self.__supportedModels.keys().__contains__(wrapperModel):
            raise Exception(ErrorMessages.AIWrapper_ModelNotSupported(wrapperModel.name))
        
        # Init clients to none
        self.__anthropicClient = None
        self.__openAIClient = None

        # Set the model of the wrapper
        self.__model = wrapperModel

        # Case the model is an OpenAI model. Set up the OpenAI client with that model.
        if (wrapperModel == AIModel.GPT_4 or 
            wrapperModel == AIModel.GPT_4_MINI or 
            wrapperModel == AIModel.O1_MINI
            ):
            self.__modelProvider = AIProvider.OPEN_AI
            self.__openAIClient = OpenAI(
                organization=self.__organizationID,
                project=self.__projectID,
                api_key=self.__openAIAPIKey
            )

            # Initialize the tokenizer
            self.__tokenizer = tiktoken.encoding_for_model(self.__supportedModels[wrapperModel])


        # Case the model is a DeepSeek model. Set up the DeepSeek client with that model
        elif (wrapperModel == AIModel.DEEP_SEEK_CHAT):
            self.__modelProvider = AIProvider.DEEP_SEEK
            self.__openAIClient = OpenAI(
                api_key=self.__deepSeekAPIKey,
                base_url=self.__deepSeekBaseURL
            )

            # Initialize the tokenizer
            self.__tokenizer = transformers.AutoTokenizer.from_pretrained( 
                os.getcwd() + '\\ModelTokenInfo\\DeepSeek\\', trust_remote_code=True
                )

        # Casethe model is an anthropic model. Set up the Anthropic client with that model.
        elif (wrapperModel == AIModel.CLAUDE_3_5_SONNET):
            self.__modelProvider = AIProvider.ANTHROPIC
            self.__anthropicClient = Anthropic(
                api_key=self.__anthropicAPIKey
            )

            # Initialize the tokenizer (n/a)


        # Initialize the conversation
        self.__currentConversation = list()
        self.__conversationTokens = 0
        self.__anthropicSystemConversation = ""
     


    #
    #   Setters & Getters
    #

    #   Conversation tuple list
    def GetConversation(self) -> list:

        retList: list = []
        # If using the anthropic model, add to the return list the system messages
        if (self.__modelProvider == AIProvider.ANTHROPIC):
            retList.append(self.__anthropicSystemConversation)
        retList.append(self.__currentConversation)
        return retList
    #   Get the last conversation in the list. Returns NONE if empty
    def GetLast(self) -> tuple:
        if len(self.__currentConversation)==0:
            return None
        return self.__currentConversation[len(self.__currentConversation)-1]
    #   Get the dictionary of available models
    def GetSupportedModels(self) -> dict:
        return self.__supportedModels
    #   Get how many tokens are within the current conversation
    def GetConversationTokenCount(self) -> int:
        return self.__conversationTokens
    
    #
    #   Methods
    #
    # Clears the current conversation
    def ClearConversation(self):
        self.__currentConversation = list()
        self.__conversationTokens = 0
        self.__anthropicSystemConversation = ""


    """ 
    #   Given:
    #       A string of text
    #       A model
    #       A conversation role
    #   This model sends the inputted string to the inputted model as the conversation role provided
    #   Returns the response the model gives
    #   *If the user wants to save the conversation, they can do so by setting "saveToConveration" to true
    #       which will generate 2 unique tuples with the input to the model and response from the model
    #       and save them to this object.
    #   *If model inputted is not supported, then an empty string is returned
    """
    def SendString(self, stringToSend: str, role: ConversationRole = ConversationRole.USER, responseTemperature: float = 0, maxTokens: int = 1024, saveToConversation: bool = False, retryAttempts: int = 3) -> str:

        # Locals
        messagesOut = []
        messageRecieved: str = ""
        newMessage = {}
        stream = ""
        syncResponse = None

        # Restrict max tokens to at least 1
        if (maxTokens < 1):
            maxTokens = 1

        # Create the conversation to send out
        messagesOut = self.__currentConversation.copy()
        newMessage = { "role": self.__supportedRoles[role], "content" : stringToSend}
        messagesOut.append(newMessage)

        #   OpenAI/DeepSeek provider case
        if (self.__modelProvider == AIProvider.OPEN_AI or self.__modelProvider == AIProvider.DEEP_SEEK):
            stream = self.__openAIClient.chat.completions.create(
                model=self.__supportedModels[self.__model],
                messages=messagesOut,
                temperature=responseTemperature,
                stream=True,
                max_completion_tokens=maxTokens
            )

            # Retrieve the response
            # If an error occurs, retry
            try:
                for buffer in stream:
                    if buffer.choices[0].delta.content is not None:
                        messageRecieved += buffer.choices[0].delta.content
            except:
                if (retryAttempts == 0):
                    Exception(ErrorMessages.AIWrapper_SendStringException(self.__model.name))
                return self.SendString(stringToSend, role, responseTemperature, maxTokens, saveToConversation, retryAttempts-1)

            # If the content is being saved, save it to the conversation list
            if saveToConversation:
                self.__currentConversation.append(newMessage)

                if messageRecieved != "" and role != ConversationRole.SYSTEM:
                    newMessage = { "role": "assistant", "content" : messageRecieved}
                    self.__currentConversation.append(newMessage)

                    # Save token count
                    self.__conversationTokens += self.StringToTokenCount(messageRecieved)

                # Save token count
                self.__conversationTokens += self.StringToTokenCount(stringToSend)

        #   Anthropic provider case
        elif (self.__modelProvider == AIProvider.ANTHROPIC):

            # If the role is system, just add the string to the system conversation string.
            try:
                if (ConversationRole.SYSTEM == role):
                    self.__anthropicSystemConversation += stringToSend
                else:
                    syncResponse = self.__anthropicClient.messages.create(
                        model=self.__supportedModels[self.__model],
                        system=self.__anthropicSystemConversation,
                        messages=messagesOut,
                        temperature=responseTemperature,
                        max_tokens=maxTokens
                    )

                    # Retrieve the response
                    messageRecieved = syncResponse.content[0].text
            except:
                if (retryAttempts == 0):
                    Exception(ErrorMessages.AIWrapper_SendStringException(self.__model.name))
                return self.SendString(stringToSend, role, responseTemperature, maxTokens, saveToConversation, retryAttempts-1)

            # If the content is being saved, save it to the conversation list
            if saveToConversation and messageRecieved != "" and role != ConversationRole.SYSTEM:
                self.__currentConversation.append(newMessage)
                newMessage = { "role": "assistant", "content" : messageRecieved}
                self.__currentConversation.append(newMessage)

                # Save token count
                self.__conversationTokens += (syncResponse.usage.input_tokens - self.__conversationTokens) + syncResponse.usage.output_tokens

        # If no response was recieved, notify.
        if messageRecieved == None:
            SystemMessages.AIWrapper_BadResponse(stringToSend)

        # Return the response
        return messageRecieved
    

    """
        Calculates roughly how many tokens an inputted string equates to based on the AIWrapper's model
    """
    def StringToTokenCount(self, string: str) -> int:
        # OpenAI case
        if (self.__modelProvider == AIProvider.OPEN_AI):
            return len(self.__tokenizer.encode(string))
        # Deep seek case
        elif(self.__modelProvider == AIProvider.DEEP_SEEK):
            return len(self.__tokenizer.encode(string))
        # Anthropic case
        elif(self.__modelProvider == AIProvider.ANTHROPIC):
            SystemMessages.AIWrapper_TokenizerNotSupported(self.__model.name)
            return 0

    #
    #   Testing/Training(lemmaList: list, PromptDictionary)
    #       Batch
    #           Get_Lemma() send and retrieve response
    #   
    #   #Loops through the data
        #Repeatedly calls get_lemma
    def tt(self, lemmaDictionary: LemmaDictionary, lemmaList: list, promptDictionary: PromptDictionary, testingBinary: bool, batchParameters: dict = {}) -> None:

        # Locals
        # testing_prompt: Prompt # Prompt used in testing
        # testing_prompt_with_example_sentence: Prompt # Prompt used in training
        # training_prompt: Prompt # Prompt used in training
        # training_prompt_with_example_sentence: Prompt # Prompt used in training
        # negative_reinforcement_prompt: Prompt # Prompt used in training to correct mistakes
        tempList: list # General purpose holding list
        tempList2: list
        # tempInt: int
        tempCV: CleanValue
        tempDict: dict
        promptStr: str
        tempStr: str
        es: str # "example sentence" holder

        # Init
        # testing_prompt_with_example_sentence = promptDictionary.GetPromptByID(8)
        # training_prompt_with_example_sentence = promptDictionary.GetPromptByID(6)
        # testing_prompt = promptDictionary.GetPromptByID(7)
        # training_prompt = promptDictionary.GetPromptByID(5)
        # negative_reinforcement_prompt = promptDictionary.GetPromptByID(9)
        tempList = []
        tempList2 = []
        # tempInt = 0
        tempDict = {}
        tempStr = ""

        # If the trainingInformation is blank and the tt() function is to train the model, initialize a default
        #   training information dictionary
        if batchParameters == {}:
            SystemMessages.AIWrapper_NoBatchParameters()
            return


        # instruction_prompt0: Prompt = promptDictionary.GetPromptByType(PromptType.INSTRUCTION)[0]
        # instruction_prompt1: Prompt = promptDictionary.GetPromptByType(PromptType.INSTRUCTION)[1]
        # instruction_prompt2: Prompt = promptDictionary.GetPromptByType(PromptType.INSTRUCTION)[2]
        # instruction_prompt3: Prompt = promptDictionary.GetPromptByType(PromptType.INSTRUCTION)[3]
        # promptStr0: str = instruction_prompt0.ProducePrompt([])
        # promptStr1: str = instruction_prompt1.ProducePrompt([])
        # promptStr2: str = instruction_prompt2.ProducePrompt([])
        # promptStr3: str = instruction_prompt3.ProducePrompt([])
        # print(f"instructions: {promptStr0, promptStr1, promptStr2, promptStr3}")
        # self.SendString(promptStr0, ConversationRole.SYSTEM, saveToConversation=True)
        # self.SendString(promptStr1, ConversationRole.SYSTEM, saveToConversation=True)
        # self.SendString(promptStr2, ConversationRole.SYSTEM, saveToConversation=True)
        # self.SendString(promptStr3, ConversationRole.SYSTEM, saveToConversation=True)

        #test: Lemma = Lemma()
        #test.GetCVsByFrequency()

        # Training
        if testingBinary==False:

            # Training (Statements)
            if batchParameters["trainingWithStatements"]:
                for lemma in lemmaList:

                    # Gather lemma centroid (closest levenshtein distance), next closest and 2 farthest away
                    tempDict = lemma.GetCVDictionary()
                    tempList = []
                    tempCV = lemma.GetCVsByFrequency()[0] # Get most common CV as seed
                    tempList2 = list(dict(sorted(tempCV.GetLevenshteinDistanceToSiblingDictionary().items(), key=lambda item: item[1])).keys()) # Gather cvs by distance ordered
                    if (len(tempList2) <= 3): # If the size is small enough, use all cvs
                        tempList.append(tempCV)
                        for cv in tempList2:
                            tempList.append(tempDict[cv])

                    else:   # Otherwise, use the closest and 2 furthest cvs
                        tempList.append(tempCV)
                        tempList.append(tempDict[tempList2[0]])
                        tempList.append(tempDict[tempList2[-1]])
                        tempList.append(tempDict[tempList2[-2]])
                    # Notably, the centroid appears first as the seed

                    for cv in tempList:

                        # Produce statement for training
                        promptStr = batchParameters["trainingPrompt"].ProducePrompt([cv.GetValue(), lemma.GetValue()])

                        # Provide the language of the prompt if the language prompt is available
                        if batchParameters["languageContextPrompt"] is not None:
                            promptStr += ' ' + batchParameters["languageContextPrompt"].ProducePrompt([lemma.GetLanguage().name])

                        # If the cv has example sentences, use the example sentence training prompt. Otherwise, use the training prompt without example sentence
                        if (cv.GetExampleSentences() != [] and batchParameters["exampleSentenceContextPrompt"] is not None):
                            es = cv.GetExampleSentences(includeMaskTokens=batchParameters["exampleSentenceMasks"])
                            promptStr += ' ' + batchParameters["exampleSentenceContextPrompt"].ProducePrompt([sklearn.utils.shuffle(es)[0]])
                        
                        self.SendString(promptStr, ConversationRole.SYSTEM, saveToConversation=True)
                        print(promptStr)
                        print(self.GetLast())
                        print("\n")

            # Training (Questions)
            if batchParameters["trainingWithQuestions"]:
                for lemma in lemmaList:

                    # Select 3 random cvs within each lemma
                    tempDict = lemma.GetCVDictionary()
                    tempList2 = list(tempDict.values())
                    if (len(tempList2) <= 5): # If the size is small enough, use all cvs
                        tempList = tempList2
                    else:
                        tempList = sklearn.utils.shuffle(tempList2)[0:5]

                    for cv in tempList:
                        
                        promptStr = ""
                        # If a correction is being made (positive or negative reinforcement), add it to the promptStr
                        if (tempStr != ""):
                            promptStr += tempStr
                            tempStr = ""

                        # Test
                        promptStr += batchParameters["testPrompt"].ProducePrompt([cv.GetValue()]) 
                        
                        # Provide the language of the prompt if the language prompt is available
                        if batchParameters["languageContextPrompt"] is not None:
                            promptStr += ' ' + batchParameters["languageContextPrompt"].ProducePrompt([lemma.GetLanguage().name])

                        # If the cv has example sentences, use the example sentence training prompt. Otherwise, use the training prompt without example sentence
                        if (cv.GetExampleSentences() != [] and batchParameters["exampleSentenceContextPrompt"] is not None):
                            es = batchParameters["exampleSentenceContextPrompt"].ProducePrompt([sklearn.utils.shuffle(cv.GetExampleSentences(includeMaskTokens=batchParameters["exampleSentenceMasks"]))[0]])
                            es = sklearn.utils.shuffle(cv.GetExampleSentences(includeMaskTokens=batchParameters["exampleSentenceMasks"]))[0].replace('\n', '')
                            promptStr += ' ' + batchParameters["exampleSentenceContextPrompt"].ProducePrompt([es])
                        else:
                            es = ""
                                

                        response = self.SendString(promptStr, ConversationRole.USER, saveToConversation=True)
                        isCorrect = True if response==lemma.GetValue() else False
                        print(promptStr)
                        print(self.GetLast())

                        # If log training or common mistake analysis, log test
                        if batchParameters["logTraining"] or batchParameters["commonMistakeCorrectionPrompt"] is not None:
                            if es != "":
                                lemmaDictionary.LogTest(lemma.GetValue(), cv.GetValue(), isCorrect, '( ' + response + ' | ' + es + ' )')
                            else:
                                lemmaDictionary.LogTest(lemma.GetValue(), cv.GetValue(), isCorrect, '( ' + response + ' )')

                        #
                        #   Positive reinforcement if response is correct
                        if (isCorrect):
                            if batchParameters["positiveReinforcementPrompt"] is not None:
                                tempStr = ""
                                tempStr = batchParameters["positiveReinforcementPrompt"].ProducePrompt([])

                        #   
                        #   Negative reinforcement if response is incorrect
                        if (not isCorrect):

                            tempStr = ""

                            # Common mistake correction
                            if batchParameters["commonMistakeCorrectionPrompt"] is not None and lemma.Tested() and batchParameters["commonMistakeFrequencyThreshold"] is not None:
                                
                                # Gather response frequency amongst all cvs tested in the lemma
                                responseFrequency = 1
                                for cv in list(lemma.GetCVDictionary().values()):
                                    if cv.Tested():
                                        for word in cv.TestIncorrectGuessWords():
                                            if word == response:
                                                responseFrequency = responseFrequency + 1

                                # If the frequency meets or exceeds the threshold, put the common mistake correction prompt
                                #   into the next string to send
                                if responseFrequency >= batchParameters["commonMistakeFrequencyThreshold"]:
                                    tempStr += ' ' + batchParameters["commonMistakeCorrectionPrompt"].ProducePrompt([lemma.GetValue(), response, lemma.GetValue()])
                                
                            # Small mistake correction
                            if batchParameters["smallMistakeCorrectionPrompt"] is not None and batchParameters["smallMistakeLevenshteinThreshold"] is not None:

                                # Get the levenshtein distance of the response to the lemma. If it is below the small mistake threshold, add the 
                                #   Small mistake prompt to the response
                                if Levenshtein.distance(response, lemma.GetValue()) <= batchParameters["smallMistakeLevenshteinThreshold"]:

                                    # Determine if it was extra distance
                                    tempStr += ' ' + batchParameters["smallMistakeCorrectionPrompt"].ProducePrompt([cv.GetValue(), lemma.GetValue()])



                            # Generic response
                            if batchParameters["negativeReinforcementPrompt"] is not None:
                                tempStr += ' ' + batchParameters["negativeReinforcementPrompt"].ProducePrompt([cv.GetValue(), lemma.GetValue()])


            # If not logging training tests, but common mistake handling was turned on, flush lemma test results in training
            if not batchParameters["logTraining"] and batchParameters["commonMistakeCorrectionPrompt"] is not None:
                lemmaDictionary.FlushTests()

        # Testing
        else:
            # For each lemma in the batch...
            for lemma in lemmaList:      
                # For each cv in each lemma...
                for cv in lemma.GetCVDictionary().values():

                    # Send testing prompt. Don't save to conversation.
                    # Test
                    promptStr = batchParameters["testPrompt"].ProducePrompt([cv.GetValue()]) 
                        
                    # Provide the language of the prompt if the language prompt is available
                    if batchParameters["languageContextPrompt"] is not None:
                        promptStr += ' ' + batchParameters["languageContextPrompt"].ProducePrompt([lemma.GetLanguage().name])

                    # If the cv has example sentences, use the example sentence training prompt. Otherwise, use the training prompt without example sentence
                    if (cv.GetExampleSentences() != [] and batchParameters["exampleSentenceContextPrompt"] is not None):
                        es = batchParameters["exampleSentenceContextPrompt"].ProducePrompt([sklearn.utils.shuffle(cv.GetExampleSentences(includeMaskTokens=batchParameters["exampleSentenceMasks"]))[0]])
                        es = sklearn.utils.shuffle(cv.GetExampleSentences(includeMaskTokens=batchParameters["exampleSentenceMasks"]))[0].replace('\n', '')
                        promptStr += ' ' + batchParameters["exampleSentenceContextPrompt"].ProducePrompt([es])
                    else:
                        es = ""

                    response = self.SendString(promptStr, ConversationRole.USER, saveToConversation=False)
                    print(promptStr)
                    print(response)

                    if response==lemma.GetValue():
                        isCorrect = True
                    else:
                        isCorrect = False
                    if es != "":
                        lemmaDictionary.LogTest(lemma.GetValue(), cv.GetValue(), isCorrect, '( ' + response + ' | ' + es + ' )')
                    else:
                        lemmaDictionary.LogTest(lemma.GetValue(), cv.GetValue(), isCorrect, '( ' + response + ' )')

            # Print accuracy after each lemma
            # print(f"Total Test Accuracy: {lemmaDictionary.TotalTestAccuracy()}")


        #print(str(training_prompt.GetParams()))
  

        return None

        # lemmaDictionary.Query([], [], [])

        # Value
        # occurences
        # .
        # .
        # .

        # filter
        # value = ""
        # occurences > 100
        
        # lemmaDictionary.GetDictionary().GetValues()
        # list
        # for each in list
        #     remove if ocrrences are less



            # intro prompt: you are this
            # training prompt: the lemma of this is this./what is the lemma of this?
            # correct prompt: correct, the lemma of this is this.
            # incorrect prompt: incorrect, the lemma of this is this.

        #send intro prompt

        #For batch #    
            # prompt = produceprompt training prompt
            # if correct & testing
            #     append in the front correct prompt
            # elif incorrect & testing
            #     append in the front incorrect prompt
            # response = get_lemma(prompt, new lemma)

            # if testing
            #     see if response was right
            #     flag correct/incorrect
            #     update accuracy and error checking
        #



    #
    #   1 sends the string and gets a response (with prompt and lemma and cv)
    #   gets the lemma out of the response from model and returns it 
    # 

    """
        Saves the current conversation in the wrapper to the cwd
    """
    def conversation_to_csv(self, fileName: str):

        conversations = self.GetConversation()

        if self.__modelProvider == AIProvider.ANTHROPIC:
            conversations = conversations[1]
        else:
            conversations = conversations[0]

        fieldnames = list(conversations[0].keys())
        with open(fileName + ".csv", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
            # Write all rows at once
            writer.writerows(conversations)


    def get_lemma(self, word: str):

        # response = self.SendString(, )
        
        # text = response.choices[0].message.content

        # lemma = extract_lemma(text)

        # return lemma
        return None
    


    """
        Given a:
            testingPrompt: prompt
            outputDirectory: str
            outputTestName: str
            Where 
        This method 
    """
    def FullTest(self, testDataFilePath: str, testingPrompt: Prompt, outputDirectory: str, outputTestName: str, debug: bool = False) -> bool:


        # Locals
        testSystemLoaded: bool = False
        testSystem: TestSystem
        currentPacket: dict = dict()
        promptString: str = ""
        response: str = ""

        # Load the test system
        testSystem = TestSystem()
        testSystemLoaded = testSystem.LoadTestFile(testDataFilePath)

        # If the TestSystem can't be loaded, return false. Notify the user.
        if (not testSystemLoaded):
            SystemMessages.AIWrapper_TestSystemFailedToLoad()
            return False
        
        # For each packet in the testData, load it. Then run all data in the packet through the AIWrapper.
        currentPacket = testSystem.NextPacket()
        while (currentPacket != {}):

            if debug:
                print("AIWrapper: FullTest - CVs to test: ")
                print(str(currentPacket["values"]))
                print('\tFrag ID:'+str(currentPacket["fragmentID"]))
                print('\tSentence Number:'+str(currentPacket["sentenceNumber"]))
                print('\tSentence:'+str(currentPacket["sentence"]))
                print('\tSentence Tokens:'+str(currentPacket["sentenceTokens"]))


            # For each value, send the value, the prompt, and context sentence if applicable to the inputted prompt.
            for cv in currentPacket["cleanValues"]:

                # Produce the prompt
                promptString = testingPrompt.ProducePrompt([cv, currentPacket["sentence"]])

                # Send the prompt, save the response
                response = self.SendString(promptString)     

                # Scrub special characters from response
                response = response.replace('\n','')
                response = response.replace('\r','')

                if debug:
                    print("AIWrapper: FullTest - CV: " + cv + " | Response: " + response)

                # Store the prompt as a prediction in the current packet
                currentPacket["predictions"].append(response)


            # Send the current packet after making a prediction on each cv
            testSystem.SendPacket(currentPacket, outputDirectory, outputTestName, append=True)

            # Print packets sent if debugging
            # if debug:
            #     print("AIWrapper FullTest: Packet sent - " + str(currentPacket))
                
            # Get the next packet
            currentPacket = testSystem.NextPacket()

        #

        # Return when finished
        return True

    

# End OpenAI module
