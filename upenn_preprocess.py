import pandas as pd
import numpy as np
import os
import dill
import pandas as pd
import glob, csv
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt') # Download this as this allows you to tokenize words in a string.

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import re
import string
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()