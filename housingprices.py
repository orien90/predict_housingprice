import csv
from okdata.sdk.data.download import Download
from tabulate import tabulate
import keras
import pandas as pd
import matplotlib.pyplot as plt

# Instantiate the download client.
client = Download()

# Download the files from the dataset's latest distribution.
res = client.download(
    dataset_id="boligpriser-blokkleiligheter",
    version="1",
    edition="latest",
    output_path="/tmp/examples",
)

# Select the first file (this is usually also the only file).
filename = res["files"][0]
# Read file contents.
with open(filename) as f:
    data = f.read()
# Print a nice table.
dialect = csv.Sniffer().sniff(data)
# print(tabulate(csv.reader(data.splitlines(), dialect)))

col_list = ['kvmpris', 'bydel_navn', 'aar']

df = pd.read_csv(filename, encoding='latin1', usecols=col_list, sep=';')
oslo = (df.loc[df['bydel_navn'].isin(['Oslo i alt'])])
# creates python list for square feet price and years in Oslo
oslo_kvm = [x[2] for x in oslo.values.tolist()]
oslo_aar = [int(x[0]) for x in oslo.values.tolist()]
# Change to pandas datafram list so it can be passed as parameters in Keras functions line 43
df_oslo_aar = pd.DataFrame(oslo_aar, columns=['aar'])
df_oslo_kvm = pd.DataFrame(oslo_kvm, columns=['kvmpris'])

# train and compile the dataset with chosen parameters. Can be changed when trying to optimize results
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile(keras.optimizers.Adam(learning_rate=1), 'mean_squared_error')
model.fit(x=df_oslo_aar, y=df_oslo_kvm, epochs=30, batch_size=10)

new_year = 2022
print("Estimert kvm pris for leiligheter i oslo 2022:")
print(model.predict([new_year]))  # likely way too small dataset for accurate prediction
