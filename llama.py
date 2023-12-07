import pandas as pd
from sklearn.model_selection import train_test_split
from time import sleep
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Load model directly
# https://huggingface.co/togethercomputer/LLaMA-2-7B-32K
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", trust_remote_code=True, torch_dtype=torch.bfloat16)

# read Cleaned_ResumeDataSet.csv into a DataFrame
df = pd.read_csv("src/data/Cleaned_ResumeDataSet.csv")

# create x_train and y_train from the DataFrame and also remove index column
x_train = df["Resume"]
y_train = df["Category"]

# random_state=42 for reproducibility


# split x_train and y_train into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, random_state=42, test_size=0.2
)

_template = "predict position of this resume: \"{}\" "

for i in range(len(x_train)):
    x_train[i] = x_train[i].replace("\n", " ")
    input_context = tokenizer.encode(_template.format(x_train[i]), return_tensors="pt")
    generated = model.generate(
        input_context,
        temperature=0.7,
        max_length=1000,
        do_sample=True,
    )
    print("Resume best fits in category: ", tokenizer.decode(generated[0], skip_special_tokens=True))