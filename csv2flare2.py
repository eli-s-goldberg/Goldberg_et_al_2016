
from pandas import *
import csv, itertools, json

# import csv, itertools, json
mycols = [
    'nmId',
    'shape',
    'nomLayer',
    'dissNomConc',
    'saltType',
    'prepMethod',
    'concPump',
    'pH',
    'temp',
    'Debye_Length',
    'depAttEff'
]
layers = ['nmId', 'shape', 'nomLayer', 'dissNomConc', 'concPump']

df = read_csv("/Users/future/PycharmProjects/javaJunk/basic_table.csv", usecols = mycols)
df.sort(mycols,inplace=True)
df.to_csv("modifiedcsv.csv", na_rep='-', header=False, index=False,
encoding='utf-8')
with open('modifiedcsv.csv', 'r') as f:
     reader = csv.reader(f)
     your_list = list(reader)

def cluster(rows):
     result = []
     for key, item in itertools.groupby(rows, lambda x: x[0]):
         group_rows = [row[1:] for row in item]

         if len(row[1:]) == 1:
                result.append({"name": row[0],"size": float(row[1])})

         else:
                result.append({"name": key,"children":cluster(group_rows)})

     return result

if __name__ == '__main__':
     rows = your_list

with open('flare.json', 'w') as outfile:
     json.dump(cluster(rows), outfile, ensure_ascii=False, indent=2)

lines = file('flare.json', 'r').readlines()
# del lines[0]
# del lines[-1]
file('flare.json', 'w').writelines(lines)

# Rember to put {
#   "name": "flare",
#   "children": [data]}
# on the end of your data!