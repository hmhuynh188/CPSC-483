# happiness and income graph 
import numpy as np
import pandas as pd
import ipywidgets as widgets

hls_all_raw = pd.read_csv('/content/drive/My Drive/Colab Notebooks/happy.csv')
print(hls_all_raw)

print(hls_all_raw["Indicator"])
print("\n===========================================================\n")
hls_slice = pd.DataFrame(hls_all_raw, columns =["Country","Indicator","Type of indicator","Time","Value"])
print(hls_slice)

hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Life satisfaction"]
print(hls_ls)
print("\n===========================================================\n")
print("Total records:")
print(len(hls_ls))

print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_ls["Country"].unique()))

print("\n===========================================================\n")
print("Country List")
print(hls_ls["Country"].unique())

hls_train = hls_ls.loc[hls_ls["Time"] == 2018]
hls_train = hls_train.loc[hls_ls["Type of indicator"] == "Average"]
print("\n===========================================================\n")
print("Total records:")
print(len(hls_train))

print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_train["Country"].unique()))

print("\n===========================================================\n")
print("Record:")
print(hls_train)

import pandas as pd
weo_raw = pd.read_csv('/content/drive/My Drive/Colab Notebooks/WEOOct2023all.csv')
na_mask = weo_raw['WEO Subject Code'].isna()
weo_selected_measurement = weo_raw.loc[~na_mask & weo_raw['WEO Subject Code'].str.contains("NGDP_RPCH")]
weo_selected_measurement_2018 = pd.DataFrame(weo_selected_measurement, columns=['Country', '2018'])
print(weo_selected_measurement_2018)

merged_train_data = pd.merge(hls_train, weo_selected_measurement_2018, on="Country")
merged_train_data = merged_train_data.rename(columns={"Value": "Happiness Measurement", "2018": "Income Measurement"})
merged_train_data = pd.DataFrame(merged_train_data, columns=['Country','Happiness Measurement', 'Income Measurement'])

print(merged_train_data)

X = np.c_[merged_train_data["Income Measurement"]]
Y = np.c_[merged_train_data["Happiness Measurement"]]
x = X.tolist()
y = Y.tolist()

# plot data
out1 = widgets.Output()
with out1:
  plt.scatter(x, y)
  plt.xlabel('Income')
  plt.ylabel('Happiness')
  plt.title("Data Plot")
  plt.show()

# fit linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, Y)

# plot predictions
predict_x = [x for x in range(901)]
predict_x = [[x/100] for x in predict_x]
predict_y = model.predict(predict_x)

out2 = widgets.Output()
with out2:
  plt.scatter(predict_x, predict_y)
  plt.scatter(x, y)
  plt.xlabel('Income')
  plt.ylabel('Happiness')
  plt.title("Prediction Line")
  plt.show()

display(widgets.HBox([out1,out2]))

# employment rate and income 
from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import pandas as pd
import ipywidgets as widgets

hls_all_raw = pd.read_csv('/content/drive/My Drive/Colab Notebooks/happy.csv')
print(hls_all_raw)

print(hls_all_raw["Indicator"])
print("\n===========================================================\n")
hls_slice = pd.DataFrame(hls_all_raw, columns =["Country","Indicator","Type of indicator","Time","Value"])
print(hls_slice)

hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Employment rate"]
print(hls_ls)
print("\n===========================================================\n")
print("Total records:")
print(len(hls_ls))

print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_ls["Country"].unique()))

print("\n===========================================================\n")
print("Country List")
print(hls_ls["Country"].unique())

hls_train = hls_ls.loc[hls_ls["Time"] == 2018]
hls_train = hls_train.loc[hls_ls["Type of indicator"] == "Average"]
print("\n===========================================================\n")
print("Total records:")
print(len(hls_train))

print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_train["Country"].unique()))

print("\n===========================================================\n")
print("Record:")
print(hls_train)

import pandas as pd
weo_raw = pd.read_csv('/content/drive/My Drive/Colab Notebooks/WEOOct2023all.csv')
na_mask = weo_raw['WEO Subject Code'].isna()
weo_selected_measurement = weo_raw.loc[~na_mask & weo_raw['WEO Subject Code'].str.contains("NGDP_RPCH")]
weo_selected_measurement_2018 = pd.DataFrame(weo_selected_measurement, columns=['Country', '2018'])

print(weo_selected_measurement_2018)

merged_train_data = pd.merge(hls_train, weo_selected_measurement_2018, on="Country")
merged_train_data = merged_train_data.rename(columns={"Value": "Employment Rate", "2018": "Income"})
merged_train_data = pd.DataFrame(merged_train_data, columns=['Country','Employment Rate', 'Income'])

print(merged_train_data)

import matplotlib.pyplot as plt
import sklearn.linear_model

X = np.c_[merged_train_data["Income"]]
Y = np.c_[merged_train_data["Employment Rate"]]
x = X.tolist()
y = Y.tolist()

# plot data
out1 = widgets.Output()
with out1:
  plt.scatter(x, y)
  plt.xlabel('Income')
  plt.ylabel('Employment')
  plt.title("Data Plot")
  plt.show()

# fit linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, Y)

# plot predictions
predict_x = [x for x in range(901)]
predict_x = [[x/100] for x in predict_x]
predict_y = model.predict(predict_x)

out2 = widgets.Output()
with out2:
  plt.scatter(predict_x, predict_y)
  plt.scatter(x, y)
  plt.xlabel('Income')
  plt.ylabel('Employment')
  plt.title("Prediction Line")
  plt.show()

display(widgets.HBox([out1,out2]))
