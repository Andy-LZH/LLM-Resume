import pandas as pd
from sklearn.model_selection import train_test_split
from time import sleep
from openai import OpenAI
import os

print(os.getenv("OPENAI_API_KEY"))
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
            "role": "system",
            "content": "You are a resume scanner that take user inputted resume content and let user know their best fitted role without any explanation. ",
        },
        {"role": "assistant", "content": "Data Scientist"},
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

# read Cleaned_ResumeDataSet.csv into a DataFrame
df = pd.read_csv("src/data/Cleaned_ResumeDataSet.csv")

# create x_train and y_train from the DataFrame and also remove index column
x_train = df["Resume"]
y_train = df["Category"]
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, random_state=42, test_size=0.2
)

print(x_test)
print(y_test)

x_test = x_test.to_list()
y_test = y_test.to_list()

dataframe = pd.DataFrame(columns=["Generated", "Actual"])
print(len(x_test))
print(x_test[0])
# iterate through x_test and print the generated text

for i in range(len(x_test)):
    x_test[i] = x_test[i].replace("\n", " ")
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a resume scanner that take user inputted resume content and let user know their best fitted role without any explanation. ",
            },
            {"role": "assistant", "content": "Data Scientist"},
            {"role": "user", "content": x_test[i]},
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    print("Resume best fits in category: ", response.choices[0].message.content)
    print("Actual category: ", y_test[i])
    print("------------------------------------------------------------")
    dataframe = dataframe._append(
        {
            "Generated": response.choices[0].message.content,
            "Actual": y_test[i],
        },
        ignore_index=True,
    )

dataframe.to_csv("./GPT4.csv", index=False)
