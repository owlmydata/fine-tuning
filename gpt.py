import json
import openai
import time
from random import sample
from typing import Optional, Dict


class FineTuning:
    def __init__(self, open_ai_key: str) -> None:
        openai.api_key = open_ai_key

    def upload_file(
        self, filename: str, 
        wait_processed: bool = True, 
        request_delay: int = 30
    ) -> str:
        """Uploads a file to OpenAI API and returns the file id.

        Args:
            filename (str): File to upload.
            wait_processed (bool): Wait until file is processed. 
                Defaults to True.
            request_delay (int): Delay between status checks in seconds. 
                Defaults to 30.
            
        Returns:
            str: The file ID.
        """
        file = openai.File.create(
            file=open(filename, "rb"), 
            purpose="fine-tune"
        )

        while wait_processed:
            status = openai.File.retrieve(file["id"])
            if status["status"] == "processed":
                print("File processed!")
                break
            else:
                print("Waiting for file to be processed...", end="\r")
                time.sleep(request_delay)

        return file["id"]

    def create_job(
        self,
        file_id: str,
        model_name: str,
        suffix: Optional[str] = None,
        hyperparameters: Dict[str, int] = {"n_epochs": 1},
    ) -> str:
        """Creates a fine-tuning job and returns the job id.

        Args:
            file_id (str): File ID for training.
            model_name (str): Model to fine-tune.
            suffix (Optional[str]): Suffix for model name. Defaults to None.
            hyperparameters (Dict[str, int]): Hyperparameters for job. 
                Defaults to {'n_epochs': 1}.
            
        Returns:
            str: The job ID.
        """
        model = openai.FineTuningJob.create(
            model=model_name,
            training_file=file_id,
            hyperparameters=hyperparameters,
            suffix=suffix,
        )
        return model["id"]

    def retrieve_when_job_is_done(
        self, job_id: str, request_delay: int = 30
    ) -> Dict[str, str]:
        """Returns the job when it is done.

        Args:
            job_id (str): Job ID to retrieve.
            request_delay (int): Delay between status checks in seconds. 
                Defaults to 30.
            
        Returns:
            Dict[str, int]: The job details.
        """
        while True:
            model = openai.FineTuningJob.retrieve(job_id)
            if model["status"] == "succeeded":
                return model
            else:
                print(f'Training status: {model["status"]}.', end="\r")
                time.sleep(request_delay)


def build_fine_tuned_model(
    open_ai_key: str,
    filename: str,
    model_name: str,
    suffix: Optional[str] = None,
    hyperparameters: Dict[str, int] = {"n_epochs": 1},
    wait_processed: bool = True,
    request_delay: int = 30,
) -> Dict[str, str]:
    """Builds a fine-tuned model and returns the job details.

    Args:
        open_ai_key (str): OpenAI API key.
        filename (str): File to upload.
        model_name (str): Model to fine-tune.
        suffix (Optional[str]): Suffix for model name. Defaults to None.
        hyperparameters (Dict[str, int]): Hyperparameters for job. 
            Defaults to {'n_epochs': 1}.
        wait_processed (bool): Wait until file is processed. Defaults to True.
        request_delay (int): Delay between status checks in seconds. 
            Defaults to 30.
        
    Returns:
        Dict[str, int]: The job details.
    """
    fine_tuning = FineTuning(open_ai_key=open_ai_key)
    file_id = fine_tuning.upload_file(
        filename=filename,
        wait_processed=wait_processed,
        request_delay=request_delay,
    )
    job_id = fine_tuning.create_job(
        file_id=file_id,
        model_name=model_name,
        suffix=suffix,
        hyperparameters=hyperparameters,
    )
    job = fine_tuning.retrieve_when_job_is_done(job_id)
    return job


def main():
    open_ai_key = '...'
    filename = "data/gpt/train.jsonl"
    model_name = "gpt-3.5-turbo"
    suffix = "pybr"
    hyperparameters = {"n_epochs": 1}
    wait_processed = True
    request_delay = 30

    model = build_fine_tuned_model(
        open_ai_key=open_ai_key,
        filename=filename,
        model_name=model_name,
        suffix=suffix,
        hyperparameters=hyperparameters,
        wait_processed=wait_processed,
        request_delay=request_delay,
    )

    print(model)

    with open(filename, 'r') as f:
        messages = [json.loads(t) for t in f.readlines()]

    for each in sample(messages, 5):
        print('-' * 200)
        print(f'System: {each["messages"][0]["content"]}')
        print(f'User: {each["messages"][1]["content"]}')

        fine_tuned_model = model['fine_tuned_model']

        completion_original = openai.ChatCompletion.create(
            model=model_name,
            messages=each['messages'][:2],
        )['choices'][0]['message']['content']

        completion_fine_tuned = openai.ChatCompletion.create(
            model=fine_tuned_model,
            messages=each['messages'][:2],
        )['choices'][0]['message']['content']

        print(f"Assistant [used-to-trained]: {each['messages'][2]['content']}")
        print(f"Assistant [GPT-3.5]: {completion_original}")
        print(f"Assistant [fine-tuned]: {completion_fine_tuned}")


if __name__ == "__main__":
    main()