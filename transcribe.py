import torch
from TTS.api import TTS
import random
import os
import pandas as pd
import gdown
import random
import time
import shutil
import tempfile
from unittest.mock import patch
import warnings
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import io
import logging
from TTS.tts.configs.xtts_config import XttsConfig

torch.serialization.add_safe_globals([XttsConfig])
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="PT2InferenceModel has generative capabilities")

# !pip install --upgrade tensorflow
# !pip install --upgrade numpy==1.22.0
# !pip install gspread
# !pip install cython>=0.29.30
# !pip install torch>=2.1
# !pip install anyascii>=0.3.0
# !pip install pyyaml>=6.0
# !pip install fsspec>=2023.6.0
# !pip install aiohttp>=3.8.1
# !pip install packaging>=23.1
# !pip install TTS
# !pip install gradio
# !pip install openpyxl
# !pip install gdown

@contextmanager
def suppress_native_output():
    with tempfile.TemporaryFile() as devnull:
        devnull_fd = devnull.fileno()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            # Redirect C-level stdout and stderr to devnull
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            yield
        finally:
            # Restore original file descriptors
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


class TEXT_TO_SPEECH:

    def __init__(self, device="cpu") -> None:
      
      device = "cuda" if torch.cuda.is_available() else "cpu"
      with patch('builtins.input', return_value='y'):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
              
      self.VOICES = { 
        'Adam': '1dJA6oL9g8OPlFtWPWebos8EJssPhcF_U',
        'Alice': '1uSorJ01N5veelDiXrOJylMoH2Iu4MjFz',
        'Aria': '1XaNapPMehklx2WqYI2oHgKtEx6NZHeD0',
        'Bill': '1oMrRsqouYEB9TjArajC_yEUVtqniPTNv',
        'George': '1OCBJR5r1OLMPLGCY1GyaQ_7zBjDUBRX8',
        'Lily': '1j2es6mPX552NwblDK0TvuP99kU8IsWdA',
        'Reachel': '1SIfGSOO1M59emMQ1xtyXUuaD9bsGNWs3',
        'Sarah': '17NAJuGqQ6xcJJNxZgxoARWzQAprNDhhx',
        "Alex": "1mzlMLVS6K_YN3dZ-pKBysX7urx7UcEtQ",
        "SARAH": "1_o2dXyX3rJlJI_9R6Vkk1XisJnvYjKZ1",
        "Realistic_Male": "1GGnXLSmR2Y2MQF-wuodie4FbAxw876mH",
        "realistic_male_2": "1bdTSj6__QMCD-5rPWh9eFfqwzhTkqc6H",
        "F5TTS_Male": "1Mngf-Q63uP1X469iCKlsELWugSE8oS8B",
        "Realistic_Man_Voice_[Short]": "13OA70vrlziGfcUPC4fSmRw9i5uTZIQgl",
        "boy": "1pmOf7Nwblu8-YCxVBgSGBwW4VeG1vvM_"
      }
      self.LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']


    def transcribe(self, message, voice, lang, output_dir=os.getcwd()):
      if lang not in self.LANGUAGES:
        print("LANGUAGE NOT FOUND... PLEASE SELECT VALID ONE")
        return None

      file_path = str(os.path.join(output_dir, f"output_voice_{random.randrange(11111, 99999)}.wav"))
      stdout_capture = io.StringIO()
      stderr_capture = io.StringIO()
      try:
        with suppress_native_output(), redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
          self.tts.tts_to_file(
              text=message,
              speaker_wav=str(voice),
              language=lang,
              file_path=file_path
        )
        return file_path
      except Exception as e:
          out = stdout_capture.getvalue()
          err = stderr_capture.getvalue()
          raise Exception(f"{out}\n{err}\n{e}")


def download_speaker(id, output_file):
  
  download_url = f'https://drive.google.com/uc?id={id}'
  gdown.download(download_url, output_file, quiet=False)
  time.sleep(2)




if __name__ == "__main__":

    TTS_OBJ = TEXT_TO_SPEECH()
    message = "Hi, How are you?"
    VOICE_TYPE = "boy"
    LANGUAGE = "en"

    speaker_voice = os.path.join(os.getcwd(), f"{VOICE_TYPE}.wav")
    if not os.path.exists(speaker_voice): download_speaker(TTS_OBJ.VOICES[VOICE_TYPE], speaker_voice)
    file_path = TTS_OBJ.transcribe(message, speaker_voice, LANGUAGE)