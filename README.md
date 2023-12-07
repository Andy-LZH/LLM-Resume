## Are we able to identify what job you are looking for based on the scanning of your resume?
#### ECS 171 Project Group 13
###### Team Leader: Zhuoheng Li
###### Team Members: Zhuoheng Li, Angelina Lim, Adham Niazi, Pramesh Sharma

## Getting Started

These instructions will help you set up and run the project on your local machine.

### Prerequisites

Make sure you have Python and virtualenv installed on your machine.

- Python (version 3.x): [Download Python](https://www.python.org/downloads/)
- virtualenv: Install using `pip install virtualenv`

### Setting up the Virtual Environment

Create a virtual environment to isolate the project dependencies.

```bash
# On Windows
python -m venv venv

# On macOS/Linux
python3 -m venv venv
```

### Activate venv

```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the project
```bash
python src/data/app.py
```

Note: accessing relative files differ depending on Mac vs. Windows
<br>
In app.py, change the paths should be reflected as such:

```bash
# Windows
model_path = 'Models/svm_model.joblib'
vectorizer_path = 'Models/tfidf_vectorizer.joblib'

# Mac
model_path = "src/data/Models/svm_model.joblib"
vectorizer_path = "src/data/Models/tfidf_vectorizer.joblib"
```
