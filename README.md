# empathic-stories

## Latest model checkpoint available here:
[Model checkpoint available for download through Google Drive](https://drive.google.com/file/d/1y6SRfWGFeW9f8AnkIWiKdRYelIRbNW1I/view?usp=drive_link) 

[Sample server code for accessing model](https://drive.google.com/file/d/1ifdX5ds2wblhBZH2IU89F6szrfLvz-WB/view?usp=drive_link) 
 

## File Structure
* `/annotation` contains all MTurk annotation templates
* `/data` contains all data folders for train, dev, test sets
* `/models` contains all lightning modules and our pretrained BART model
    * `EmpathicSimilarityModel` takes in a story pair (2 stories) and fine tunes on empathic similarity score
    * `EmpathicSummaryModel` takes in a single story and fine tunes on empathy reasons (main event + emotion description + moral)
* `/config` contains yaml config files for different model training settings
* `/user_study` contains the frontend and server side code for our user study interface
* `dataset.py` contains the dataloaders
* `special_tokens.py` definitions of special tokens
* `trainer.py` contains training code and input of config files for different model training settings
* `utils.py` contains extra model utilities
* `evaluator.py` contains an evaluation class to compute all evaluation metrics

## Dataset Overview
### Stories
* `Data Source`: which data source the story came from
* `story`: raw text of the story
* `story_formatted`: the story formatted with breaks
* `story_summary`: ChatGPT summarized story
* `comments`: (if pulled from social media), top level comments to the story
* `url`: (if pulled from social media), the original url of the story
* `post_id`: (if pulled from social media), the original id of the story
* `post_time`: (if pulled from social media), the time the story was posted
* `post_score`: (if pulled from Reddit), the score of the post
* `toxicity_score`: toxicity score rated by Detoxify
* `WorkerId`: worker ID of annotator
* `LifetimeApprovalRate`: annotator's lifetime approval rate
* `AcceptTime`: when the annotator accepted the HIT
* `SubmitTime`: when the annotator submitted the HIT
* `WorkTimeInSeconds`: how long the annotator took for the HIT
* `Age`: annotator age
* `Gender`: annotator gender
* `Race`: annotator race
* `Arousal`: annotator's arousal before the task (1-10)
* `Valence`: annotator's valence before the task (1-10)
* `Main Event`: main event of the story as rated by human annotator
* `Emotion Description`: emotion of the story as rated by human annotator
* `Moral`: moral of the story as rated by human annotator
* `Empathy Reasons`: reasons why people may empathize with the story as rated by human annotator
* `Main Event (gpt3.5)`: main event of the story as rated by ChatGPT
* `Emotion Description (gpt3.5)`: emotion of the story as rated by ChatGPT
* `Moral (gpt3.5)`: moral of the story as rated by ChatGPT
* `Empathy Reasons (gpt3.5)`: reasons why people may empathize with the story as rated by ChatGPT
* `Empathizable`: how generally "empathizable" the story is
* `Well-Written`: how well-written the story is
* `fake_score`: how likely the post is written by AI tools, as predicted by the Writer AI Content Detector
* `num_sentences`: number of sentences in the story
* `num_words`: number of words in the story
* `num_sentences_event`: number of sentences in the event
* `num_words_event`: number of words in the event
* `num_sentences_emotion`: number of sentences in the emotion
* `num_words_emotion`: number of words in the emotion
* `num_sentences_moral`: number of sentences in the moral
* `num_words_moral`: number of words in the moral
* `num_sentences_empathy_reasons`: number of sentences in the empathy reasons
* `num_words_empathy_reasons`: number of words in the empathy reasons


### Story Pairs
* `pairs`: pair ID (matches with story file index)
* `binned`: which sampled bin the pair belongs to (based on SBERT sampling)
* `story_A`: first story in story pair
* `story_B`: second story in story pair
* `story_A_summary`: summary of first story in story pair
* `story_B_summary`: summary of second story in story pair
* `Empathic Similarity (gpt3.5)`: empathic similarity score as rated by ChatGPT
* `Empathic Similarity Binned (gpt3.5)`: binned empathic similarity score as rated by ChatGPT
* `Empathic Similarity Reasons (gpt3.5)`: reasons why two stories are empathically similar as rated by ChatGPT
* `similarity_empathy_human_AGG`: empathic similarity score as rated by human annotators
* `similarity_event_human_AGG`: event similarity score as rated by human annotators
* `similarity_emotion_human_AGG`: emotion similarity score as rated by human annotators
* `similarity_moral_human_AGG`: moral similarity score as rated by human annotators
