# EvaCunWWURepository
Code repository for the WWU Computer Science Department during their participation in EvaCun 2025. https://digitalpasts.github.io/EvaCUN/

Our project uses existing LLMs and Prompt Engineering in order to lemmatize data. To set up the system:
1) Download VSCode. Open the folder "EvaCunSubmissionRepo\Prompt Engineering Project\" on VSCode once cloned onto your device.
  https://code.visualstudio.com/download
2) Install Python 3.12.8 on your device
   https://www.python.org/downloads/
4) Set up a python interpreter and environment within VSCode. The target version is 3.12.8
   https://code.visualstudio.com/docs/python/python-tutorial
5) Install the necessary python libraries for the system using pip (comes with python 3.12.8). We use:
   - tiktoken
   - scikit-learn
   - numpy
   - pandas
   - enum
   - typing
   - os
   - Levenshtein
   - events
   - transformers
   - time
   - openai
   - anthropic
   - csv
   - sys

   Open a terminal within the "EvaCunSubmissionRepo\Prompt Engineering Project" folder, for each library mentioned, type the following command and replace [LIBRARY] with the library name:

   py -m pip install [LIBRARY]

6) From this point, the project is setup, but the model can't run until you give it an API key depending on the model you want to use.

   For OpenAI models, (gpt-4o, gpt-4-mini, deepseek, etc):
   a) Go to "AIModule.py"
   b) For using openai models, create an openAI organization, create a project within that organization, then create an API key within that project. Enter the project ID for your project on line 73 for variable __projectID, organization ID on line 74 for variable __organizationID, and api key on line 76 for variable __openAIAPIKey.
   
   - Organization ID reference: https://platform.openai.com/account/organization
   - Project ID reference: https://help.openai.com/en/articles/9186755-managing-projects-in-the-api-platform
   - API key reference: https://platform.openai.com/docs/guides/production-best-practices#api-keys

  c) In main, on line 44 in "main.py" for variable "modelUsing" change the enum to any among AIModel.GPT_4, AIModel.GPT_4_MINI, AIModel.O1_MINI. These are the supported models for our system.
  
7) Now the model is prompted based on the contents of "main.py"
   
   - line 69 - instructionPrompts: Specify which prompts and the order of prompts to be used as instructions for the model. Use "promptDictionary.GetPromptByID()" with an ID as input to specify the prompt. Prompts along with their IDs seen in "EvaCunSubmissionRepo\Datasets\Prompt Engineering Data Design Prototype.csv"

   - line 75 - batchParameters: Specify how ICL should be done.
     - logTraining: bool, whether to save a log of ICL results with ICL conversation.
     - exampleSentenceMasks: bool, whether in prompting when providing example sentences as additional context to fill missing words in a sentence with [MASK] or not.
     - trainingWithQuestions: bool, whether to train with question prompts
     - trainingPrompt: Prompt (class), statment prompt
     - trainingWithStatements: bool, whether to train with statement prompts
     - testingPrompt: Prompt (class), question prompt
     - exampleSentenceContextPrompt: Prompt (class) or None, prompt used to provide example sentence context for a clean value used in statement or question prompting. If set to None, no example sentence is given as context for statement and question prompts.
     - languageContextPrompt: Prompt (class) or None, prompt used to provide language context for a clean value used in statement or question prompting. If set to None, no language is given as context for statement and question prompts.
     - negativeReinforcementPrompt: Prompt (class) or None, prompt used for negative reinforcement prompting. If set to None, no negative reinforcement is provided when in ICL the model provides an incorrect answer.
     - smallMistakeCorrectionPrompt: Prompt (class) or None, prompt used for small mistake correcion. If set to None, no negative reinforcement small mistake correction is provided to the model during ICL.
     - smallMistakeLevenshteinThreshold: integer, threshold for model response during question prompting in ICL to be considered a small mistake. Distance between the response (lemma guess) and the correct answer (actual lemma) is calculated as edit distance https://en.wikipedia.org/wiki/Edit_distance. If edit distance between the response is equal to or less than the threshold, the smallMistakeCorrectionPrompt is sent.
     - commonMistakeCorrectionPrompt: Prompt (class) or None, prompt used for common mistake correcion. If set to None, no negative reinforcement common mistake correction is provided to the model during ICL.
     - commonMistakeFrequencyThreshold: integer, threshold for model response during question prompting in ICL to be considered a common mistake. If the model's response as a guess for the lemma of a clean value is a common guess within all model guesses of that lemma's clean values, determined by that response's appearance across all of that lemma's clean values, then the common mistake correction prompt is sent.
     - positiveReinforcementPrompt: Prompt (class) or None, prompt used for positive reinforcement prompting. If set to None, no negative reinforcement is provided when in ICL the model provides a correct answer.
   
  - line 141 - testBatch: Specify test batch size (change [0:30] to [0:N] where N is the size of the desired test batch. Batch is formed from the training set (random 20% subset of all lemmas)).

  - lines 147 to 156 - training Batch: Specify what batches appear as training batches. Batch is formed from the dev set (random 80% subset of all lemmas)

  For additional context, batches are created from the file "Evacun_out_new.csv" located in the "Prompt Engineering Project" folder. This file contains lemmas, their clean values, and statistical data on each of them. 
  Create batch using method 
  "lemmaDictionary.LemmaQuery()" in the form of:
  lemmaDictionary.LemmaQuery(
        filters=[],
        orderBy=[],
        )[0:N]
  Where:
    - filters is a list of pandas formatted filters applied to to the training set. Column names are as they appear in "EvaCun_out_new.csv"
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.filter.html
    - orderby is a list of pandas formatted orderby arguments applied to the training set. Column names are as they appear in "EvaCun_out_new.csv"
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
    - N represents the batch size

7) After specifying parameters and creating testing and training batches, the project is ready to execute.
  
