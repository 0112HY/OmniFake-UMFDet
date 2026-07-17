# import os
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset

# PROMPT_TEMPLATE = (
#     "Task: Multimodal Misinformation Classification.\n"
#     "News Headline: {new_title}\n"
#     "Given an image and its associated text, classify the sample into one of the following categories:"
#     "- real: the image and text are authentic and consistent."
#     # "- rumor: the content is false, misleading, or semantically inconsistent, but no explicit manipulated region is required in the output."
#     "- manipulated: the image or the text has been deliberately manipulated."
#     "Answer:"
# )
# class FakeDataset(Dataset):
#     def __init__(self, csv_file_path: str, image_directory_path: str):
#         """
#         初始化CSV数据集，加载CSV文件并根据`id`加载相应的图像
#         Args:
#             csv_file_path (str): CSV文件的路径
#             image_directory_path (str): 存放图像的文件夹路径
#         """

#         self.data = pd.read_csv(csv_file_path)
#         self.image_directory_path = image_directory_path
#         self.task_prompt = "<FakeVQA>"

#     def __len__(self):
        
#         return len(self.data)

#     def __getitem__(self, idx):
#         """
#         获取指定索引的数据项，包括图像和文本
#         Args:
#             idx (int): 数据索引
#         """
#         row = self.data.iloc[idx]


#         #dgm4
#         text_label = row['fake_cls']  
#         new_title = row['text']  
#         # label =  row['label']

#         root_path = "/data1/lihy/datasets"
#         image_path = os.path.join(root_path, row['image'])
#         # image_path =  row['image']
#         answers = text_label
#         prompt = PROMPT_TEMPLATE.format(new_title=new_title)
        
#         question = self.task_prompt  + prompt
#         try:
#             image = Image.open(image_path).convert("RGB") 
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Image file {image_path} not found.")

#         return question,answers,image



import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# PROMPT_TEMPLATE = "Classify the authenticity and category of this news. Title: {}"

# PROMPT_TEMPLATE = """<DETECTION_NEWS> Task: Multimodal Fake news detection. 
# Analyze the image and title. Classify the authenticity (Auth: real/fake) and the manipulation types (Vis, Txt, Cat).
# Definitions of 5 Categories (Cat):
# - real: Authentic news. Untampered image and human-written text.
# - rumor: The title contains misleading information or false claims (Misleading).
# - vision_manipulation: Image is altered (e.g., AI_Generated_Image, Image_Inpainting, Face_Swap, Face_Attribute_Editing), but text is authentic.
# - text_manipulation: Title is altered (e.g., Keyword_Replacement, AI_Generated_Text), but image is authentic.
# - mixed_manipulation: Both the image and the text are altered.
# News Title: {}"""


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
# PROMPT_TEMPLATE = "The following are multiple choice questions about fake news detection. The news headline is: {new_title}. Based on prior knowledge (e.g., historical facts, common knowledge, publicly available information sources, etc.), please carefully examine the features of the entire news image, facial features, specific areas of the image, and the news headline. If the image is generated or if there are clear signs of manipulation in the image (such as unnatural backgrounds, unrealistic facial features (e.g., emotion editing, face-swapping), image stitching, or partial modifications), and if the news headline also shows signs of modification, then the news is categorized as Manipulated. If the image and text have not been manipulated, and the semantic relationship and emotional expression between the image and text are consistent, and these details align with reality, then the news is categorized as Real. If the semantic relationship or emotional expression between the image and text is inconsistent, or if the information does not align with factual reality and lacks verifiable evidence, then the news is categorized as Rumor. Question: What category does this news belong to? A. Real. B. Rumor. C. Manipulated. The answer is:"
class FakeDataset(Dataset):
    def __init__(self, csv_file_path: str, image_directory_path: str):
        """
        初始化CSV数据集，加载CSV文件并根据`id`加载相应的图像
        Args:
            csv_file_path (str): CSV文件的路径
            image_directory_path (str): 存放图像的文件夹路径
        """


        self.data = pd.read_csv(csv_file_path)
        self.image_directory_path = image_directory_path
        self.task_prompt = "<DETECTION_NEWS>"


    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项，包括图像和文本
        Args:
            idx (int): 数据索引
        """
        row = self.data.iloc[idx]

        news_title = row['news_title']
        answers = row['label']
        img_path = row['img_path']
        label_5 = row['label_5']
        # prompt = PROMPT_TEMPLATE.format(new_title=news_title)
        prompt = PROMPT_TEMPLATE.format(news_title)

        question = self.task_prompt + prompt

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")

        # 返回图像和文本标签（标签可以根据需要进行修改）
        return question, answers, image