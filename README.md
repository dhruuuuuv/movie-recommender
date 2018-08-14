# movie-recommender
Using Python and the Movielens datasets to build a movie recommmender, that given a user and a set of movies, and other user, can produce a ranked list of recommended movies.

# DETAILS


The program is written using python 3, so assuming that python3 is installed on your machine, the whole program can be run by executing:

    python mr.py

after unzipping the files. This assumes that the data is in a file called ‘u.data’ and it is in the root of the folder.

# DEPENDENCIES

The python program uses the following dependencies:

    import csv
    import math
    import operator
    import collections

    import numpy as np

where numpy may need to be installed separately. This can be done using pip or conda.

# FILE STRUCTURE

The program is in the ‘mr.py’ file. The latex template can be compiled using the tex file inside the ‘latex’ folder.
