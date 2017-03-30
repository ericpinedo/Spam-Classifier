import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#reads all the files in a path and saves the body of a file
def readFiles(path):
    
    #iterates through every single file in a director using the os.walk
    # to find files in a direcotry 
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            
            # saves the path for the file in the directory
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            
            # appends all the lines but the header line 
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
           
            # close the file
            f.close()
            
            # saves the lines of the file to message
            message = '\n'.join(lines)
            
            yield path, message

# appends a new file to the dataFrame
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

# saves the emails from a local directory 
data = data.append(dataFrameFromDirectory('./emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('./emails/ham', 'ham'))

#preview of the the tablular data to determine if the emailes were correctly appeneded 
data.head()

#tokenizes all the words in the email file and counts the number of times 
# a word ocurs in a given email file
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

#MultinomialNB function from scikit learn to perfrom naive bayes on data
classifier = MultinomialNB()

#classification data for each email
targets = data['class'].values

#creates the model to predict if emails are spam based on the data given 
classifier.fit(counts, targets)

examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]

#tokenizes the words given in the list above 
example_counts = vectorizer.transform(examples)

#uses the predict function to determine if the 
predictions = classifier.predict(example_counts)
 print (predictions)



