import pandas as pd
from sklearn.model_selection import train_test_split

# BANKING, SNIPS and CLINIC raw data have the same format, so you could use this file to handle them each.

# Read CSV file into a DataFrame
name = './G4_BANKING77'
file_path = name+'.csv'
output_train_path = name+'_train.csv'
output_test_path = name+'_test.csv'
data = pd.read_csv(file_path)

# Assume 'target' is the column you want to predict
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target variable (last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split results to CSV files
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv(output_train_path, index=False)
test_data.to_csv(output_test_path, index=False)
