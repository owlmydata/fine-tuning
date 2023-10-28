import openai
import time
from typing import Optional, Dict


class FineTuning:
    def __init__(self, open_ai_key: str) -> None:
        openai.api_key = open_ai_key

    def upload_file(
        self,
        filename: str,
        wait_processed: bool = False,
        request_delay: int = 30,
    ) -> str:
        """Uploads a file to OpenAI API and returns the file id.
        
        Args:
            filename (str): The name of the file to upload.
            wait_processed (bool): Whether to wait until the file is processed. Defaults to False.
            request_delay (int): The delay between status check requests in seconds. Defaults to 30.
            
        Returns:
            str: The file ID.
        """
        file = openai.File.create(
            file=open(filename, "rb"), purpose="fine-tune"
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
            file_id (str): The ID of the file to use for training.
            model_name (str): The name of the model to fine-tune.
            suffix (Optional[str]): A suffix to append to the model name. Defaults to None.
            hyperparameters (Dict[str, int]): A dictionary of hyperparameters for the fine-tuning job. Defaults to {'n_epochs': 1}.
            
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
    ) -> Dict[str, int]:
        """Returns the job when it is done.
        
        Args:
            job_id (str): The ID of the job to retrieve.
            request_delay (int): The delay between status check requests in seconds. Defaults to 30.
            
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