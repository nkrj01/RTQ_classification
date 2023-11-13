import regex
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt

df = pd.read_excel(r"C:\Users\nraj01\Downloads\records_as_of_2023_07_15_PDT (1).xlsx")
df = df[["Question Text"]]
df = df.reset_index(drop=True)
df["clean text"] = df["Question Text"]

wc = WordCloud(width=5000,
               height=3000,
               random_state=1,
               background_color="white",
               colormap="Blues",
               collocations=False,
               stopwords=STOPWORDS).generate(''.join(df["clean text"]))

plt.figure(figsize=(15, 10))
plt.imshow(wc)