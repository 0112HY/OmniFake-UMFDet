
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


PROMPT_TEMPLATE = """Task: Multimodal Fake News Classification.
News Title: {}.
Instruction: Analyze the provided image and news title. Categorize the content into EXACTLY ONE of the following 5 categories:
Definitions:
- real: Authentic news. Both the image and text are entirely untampered and human-created.
- rumor: The image is authentic, but the title contains misleading information, false claims, or out-of-context text.
- vision_manipulation: The image is manipulated while the text remains authentic. Visual manipulation explicitly includes: AI-generated images, image inpainting, face swapping, or face attribute editing.
- text_manipulation: The image is authentic, but the text has been altered (e.g., keyword replaced or AI-generated).
- mixed_manipulation: Both the image and the text have been manipulated. (e.g., The image contains AI generation/inpainting/face swapping/attribute editing, AND the text is also altered).
The News belongs to:
"""
class FakeDataset(Dataset):
    def __init__(self, csv_file_path: str, image_directory_path: str):



        self.data = pd.read_csv(csv_file_path)
        self.image_directory_path = image_directory_path
        self.task_prompt = "<DETECTION_NEWS>"


    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        news_title = row['news_title']
        answers = row['label']
        img_path = row['img_path']
        label_5 = row['label_5']
        prompt = PROMPT_TEMPLATE.format(news_title)

        question = self.task_prompt + prompt

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")

        return question, answers, image
