# Content-Based Movie Recommender System

This project implements a **content-based recommendation engine** using the **Tag Genome 2021** dataset. It calculates movie similarities based on user-applied tags using TF-IDF and Cosine Similarity to predict ratings and recommend movies.

## 📂 Data Setup (Required)

Due to file size limits, the dataset is not included in this repository. You must download and place the files manually.

1.  **Download the dataset:**
    [https://files.grouplens.org/datasets/tag-genome-2021/genome_2021.zip](https://files.grouplens.org/datasets/tag-genome-2021/genome_2021.zip)

2.  **Create a `data` folder:**
    Create a folder named `data` in the root directory of this project.

3.  **Move the files:**
    Extract the zip file and move the following 3 files into the `data` folder:
    * `ratings.json`
    * `tags.json`
    * `tag_count.json`

**Your folder structure should look like this:**

```text
recommender-system/
├── data/
│   ├── ratings.json
│   ├── tags.json
│   └── tag_count.json
├── src/
│   └── main.py
└── README.md
