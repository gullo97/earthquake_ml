This notebook contains the code for preprocessing data from the series of earthquakes that occurred near the city of L'Aquila in Italy in 2008. The data contains info about the location and charateristics of buildings in the area, as well as the damage reported. This is quite messy data, so we will need to do a lot of cleaning and processing to get it into a usable form. 

First of all, we need to import the necessary libraries. We will be using pandas for data manipulation and numpy for numerical operations, among others. 


```python
import pandas as pd
from tqdm import tqdm
from nltk import flatten
import numpy as np
import matplotlib.pyplot as plt
import copy
```

Let's start by importing the data and taking a look 


```python
dataframe =  pd.read_excel('aquila.xls')
dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>identificativo</th>
      <th>numerorichiesta</th>
      <th>codiceaggregatostrutturale</th>
      <th>codiceedificiostrutturale</th>
      <th>comune</th>
      <th>codicecomune</th>
      <th>localita</th>
      <th>codicelocalita</th>
      <th>identificativoposizioneedificio</th>
      <th>coordinate_lat</th>
      <th>...</th>
      <th>sez3_identificativocopertura</th>
      <th>sez4_danno_strutturale_strutture_verticali</th>
      <th>sez4_danno_strutturale_solai</th>
      <th>sez4_danno_strutturale_scale</th>
      <th>sez4_danno_strutturale_copertura</th>
      <th>sez4_danno_strutturale_tamponature_tramezzi</th>
      <th>sez5_danno_elementi_non_strutturali</th>
      <th>sez6_pericolo_esterno</th>
      <th>sez7_dissesti_geologico_tecnici</th>
      <th>sez7_morfologia_versante</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>1.106610e+16</td>
      <td>0010099999</td>
      <td>NaN</td>
      <td>TIONE DEGLI ABRUZZI</td>
      <td>66100.0</td>
      <td>TIONE DEGLI ABRUZZI</td>
      <td>66100.0</td>
      <td>Angolo</td>
      <td>42203395.0</td>
      <td>...</td>
      <td>Non spingente pesante</td>
      <td>Danno Nullo</td>
      <td>Danno Nullo</td>
      <td>Danno Nullo</td>
      <td>Danno Nullo</td>
      <td>Danno Nullo</td>
      <td>NaN</td>
      <td>Assente</td>
      <td>Assenti</td>
      <td>Pendio leggero</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>1.106800e+16</td>
      <td>6800599999</td>
      <td>NaN</td>
      <td>BUSSI SUL TIRINO</td>
      <td>68005.0</td>
      <td>BUSSI SUL TIRINO</td>
      <td>680051002.0</td>
      <td>NaN</td>
      <td>422135849.0</td>
      <td>...</td>
      <td>Non spingente leggera</td>
      <td>Danno D1 Leggero:&lt;1/3</td>
      <td>Danno D1 Leggero:&lt;1/3</td>
      <td>Danno Nullo</td>
      <td>Danno Nullo</td>
      <td>Danno D1 Leggero:&lt;1/3</td>
      <td>NaN</td>
      <td>Assente</td>
      <td>Assenti</td>
      <td>Pendio leggero</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>1.106604e+16</td>
      <td>6604302213</td>
      <td>001</td>
      <td>FONTECCHIO</td>
      <td>66043.0</td>
      <td>FONTECCHIO</td>
      <td>66043.0</td>
      <td>Angolo</td>
      <td>422299396.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>Danno D4-D5 Gravissimo:&lt;1/3</td>
      <td>Danno D4-D5 Gravissimo:&lt;1/3</td>
      <td>Danno Nullo</td>
      <td>Danno Nullo</td>
      <td>Danno Nullo</td>
      <td>Presente</td>
      <td>Presente</td>
      <td>Assenti</td>
      <td>Pendio forte</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000</td>
      <td>1.106609e+16</td>
      <td>6608700976</td>
      <td>003</td>
      <td>SAN DEMETRIO NE' VESTINI</td>
      <td>66087.0</td>
      <td>SAN DEMETRIO NE' VESTINI</td>
      <td>66087.0</td>
      <td>Interno</td>
      <td>422888836.0</td>
      <td>...</td>
      <td>Spingente pesante</td>
      <td>Danno D1 Leggero:&lt;1/3</td>
      <td>Danno D1 Leggero:&lt;1/3</td>
      <td>Danno Nullo</td>
      <td>Danno Nullo</td>
      <td>Danno Nullo</td>
      <td>Presente</td>
      <td>Assente</td>
      <td>Assenti</td>
      <td>Pianura</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10001</td>
      <td>1.106609e+16</td>
      <td>0008700271</td>
      <td>001</td>
      <td>SAN DEMETRIO NE' VESTINI</td>
      <td>66087.0</td>
      <td>SAN DEMETRIO NE' VESTINI</td>
      <td>660871001.0</td>
      <td>Interno</td>
      <td>422892088.0</td>
      <td>...</td>
      <td>Non spingente leggera</td>
      <td>Danno D4-D5 Gravissimo:&lt;1/3, Danno D2-D3 Medio...</td>
      <td>Danno D4-D5 Gravissimo:&lt;1/3, Danno D1 Leggero:...</td>
      <td>Danno Nullo</td>
      <td>Danno D1 Leggero:&lt;1/3</td>
      <td>Danno Nullo</td>
      <td>Presente</td>
      <td>Assente</td>
      <td>Assenti</td>
      <td>Pendio leggero</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



As we can see, this is quite messy. There are a lot of missing values, and the data is not in a usable form. We will need to do some cleaning and processing to get it into a usable form. Moreover, many columns are not needed for our analysis, so we will drop them. Let's start by converting the coordinates to a usable form.


```python
coordinates_columns = ['coordinate_lat', 'coordinate_lon']
def coord_to_float(coord):
    coord = str(coord)
    if coord != 'nan':
        #split after dot and take first element
        coord = coord.split('.')[0]
        # add a dot after second number
        coord = coord[:2] + '.' + coord[2:]
        return float(coord)
    else:
        return float(coord)
    
for column in coordinates_columns:
    dataframe[column] = dataframe[column].apply(coord_to_float)
    #drop rows with 0.0 coordinates
    dataframe = dataframe.dropna(subset=[column])
    #normalize coordinates
    mean_coord = dataframe[column].mean()
    std_coord = dataframe[column].std()
    dataframe[column] = (dataframe[column] - mean_coord) / std_coord
    print(dataframe[column].head())
```

    0   -0.867421
    1   -0.789493
    2   -0.664419
    3   -0.213638
    4   -0.211151
    Name: coordinate_lat, dtype: float64
    0    0.596896
    1    1.602386
    2    0.440534
    3    0.191047
    4    0.169190
    Name: coordinate_lon, dtype: float64
    

Now that we have the coordinates in a usable form, we drop the columns that we don't need and reorganize the data into xs and ys.


```python
x_columns = ['coordinate_lat', 
    'coordinate_lon',
    'identificativoposizioneedificio',
    # 'sez3_regolarita2',
    # 'sez3_rinforzata',
    'sez3_struttura_orizzontale_1',
    'sez2_altezzamediapiano',
    'sez2_pianiinterrati',
    # 'sez3_mista',
    'sez3_struttura_verticale_1',
    'sez2_numeropiani',
    # 'sez3_catene_o_cordoli_1',
    'sez2_superficiepiano',
    # 'sez3_regolarita1',
    'sez3_pilastriisolati',
    # 'sez3_struttura_verticale_2',
    'sez2_costruzioneristrutturazione1',
    # 'sez3_catene_o_cordoli_2',
    # 'sez3_struttura_orizzontale_2',
    # 'sez2_costruzioneristrutturazione2',
    'sez7_morfologia_versante'
    ]
y_columns = ['sez4_danno_strutturale_strutture_verticali',
    'sez4_danno_strutturale_scale',
    'sez4_danno_strutturale_tamponature_tramezzi',
    'sez4_danno_strutturale_copertura',
    'sez4_danno_strutturale_solai'
    ]
y_columns = ['sez4_danno_strutturale_scale']

Y_SIZE = len(y_columns)
# reorganize dataframe ##########################################################
dataframe_x = dataframe[x_columns]
dataframe_y = dataframe[y_columns]
dataframe = pd.concat([dataframe_x, dataframe_y], axis=1)
# print(len(dataframe.columns))
dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coordinate_lat</th>
      <th>coordinate_lon</th>
      <th>identificativoposizioneedificio</th>
      <th>sez3_struttura_orizzontale_1</th>
      <th>sez2_altezzamediapiano</th>
      <th>sez2_pianiinterrati</th>
      <th>sez3_struttura_verticale_1</th>
      <th>sez2_numeropiani</th>
      <th>sez2_superficiepiano</th>
      <th>sez3_pilastriisolati</th>
      <th>sez2_costruzioneristrutturazione1</th>
      <th>sez7_morfologia_versante</th>
      <th>sez4_danno_strutturale_scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.867421</td>
      <td>0.596896</td>
      <td>Angolo</td>
      <td>Volte senza catene</td>
      <td>2.50-3.50</td>
      <td>0</td>
      <td>Muratura buona qualità</td>
      <td>2</td>
      <td>50-70</td>
      <td>no</td>
      <td>&lt;1919</td>
      <td>Pendio leggero</td>
      <td>Danno Nullo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.789493</td>
      <td>1.602386</td>
      <td>NaN</td>
      <td>Volte senza catene</td>
      <td>2.50-3.50</td>
      <td>Non compilato</td>
      <td>Muratura cattiva qualità</td>
      <td>4</td>
      <td>70-100</td>
      <td>no</td>
      <td>&lt;1919</td>
      <td>Pendio leggero</td>
      <td>Danno Nullo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.664419</td>
      <td>0.440534</td>
      <td>Angolo</td>
      <td>Volte senza catene</td>
      <td>2.50-3.50</td>
      <td>0</td>
      <td>Muratura cattiva qualità</td>
      <td>3</td>
      <td>50-70</td>
      <td>no</td>
      <td>&lt;1919</td>
      <td>Pendio forte</td>
      <td>Danno Nullo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.213638</td>
      <td>0.191047</td>
      <td>Interno</td>
      <td>Travi con soletta rigida</td>
      <td>2.50-3.50</td>
      <td>Non compilato</td>
      <td>Non identificata</td>
      <td>2</td>
      <td>&lt;50</td>
      <td>no</td>
      <td>1992-2001</td>
      <td>Pianura</td>
      <td>Danno Nullo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.211151</td>
      <td>0.169190</td>
      <td>Interno</td>
      <td>Volte senza catene</td>
      <td>2.50-3.50</td>
      <td>0</td>
      <td>Muratura cattiva qualità</td>
      <td>2</td>
      <td>130-170</td>
      <td>no</td>
      <td>&lt;1919</td>
      <td>Pendio leggero</td>
      <td>Danno Nullo</td>
    </tr>
  </tbody>
</table>
</div>



In this form, the data is much easier to work with. The last Y_SIZE columns contain the damage values for each building. In this case we choose to study only one type of damage (damage to staircases structures), so we will drop the other columns.

We now count the nans in each column: 


```python
for column in dataframe.columns:
    print(column, dataframe[column].isnull().sum())
```

    coordinate_lat 0
    coordinate_lon 0
    identificativoposizioneedificio 2858
    sez3_struttura_orizzontale_1 12285
    sez2_altezzamediapiano 0
    sez2_pianiinterrati 0
    sez3_struttura_verticale_1 0
    sez2_numeropiani 0
    sez2_superficiepiano 0
    sez3_pilastriisolati 0
    sez2_costruzioneristrutturazione1 1880
    sez7_morfologia_versante 1454
    sez4_danno_strutturale_scale 11
    

As almost all of the data is categorical, we save all the unique values for each column in a dictionary.


```python
unique_values_dict = {}
for column in dataframe.columns:
    if column in coordinates_columns:
        continue
    column_uniques = dataframe[column].unique() if type(dataframe[column]) == pd.Series else dataframe[column].iloc[:,0].unique()
    unique_values_dict[column] = column_uniques
# %% display unique values for each column and save as xls
print(unique_values_dict)
```

    {'identificativoposizioneedificio': array(['Angolo', nan, 'Interno', 'Isolato', 'Estremità'], dtype=object), 'sez3_struttura_orizzontale_1': array(['Volte senza catene', 'Travi con soletta rigida',
           'Travi con soletta semirigida', 'Travi con soletta deformabile',
           nan, 'Volte con catene', 'Non identificata'], dtype=object), 'sez2_altezzamediapiano': array(['2.50-3.50', 'Non compilato', '3.50-5.0', '<2.50', '>5.0'],
          dtype=object), 'sez2_pianiinterrati': array([0, 'Non compilato', 1, 2, '>2'], dtype=object), 'sez3_struttura_verticale_1': array(['Muratura buona qualità', 'Muratura cattiva qualità',
           'Non identificata',
           'Struttura Mista telaio-muratura di buona qualità',
           'Telai in c.a.',
           'Struttura Mista telaio-muratura di cattiva qualità',
           'Pareti in c.a.', 'Telai in acciaio'], dtype=object), 'sez2_numeropiani': array([2, 4, 3, 1, 5, 6, 'Non compilato', 8, 12, 7, 11, 9, 10],
          dtype=object), 'sez2_superficiepiano': array(['50-70', '70-100', '<50', '130-170', '100-130', 'Non compilato',
           '170-230', '400-500', '230-300', '300-400', '900-1200', '500-650',
           '1200-1600', '650-900', '1600-2200', '>3000', '2200-3000'],
          dtype=object), 'sez3_pilastriisolati': array(['no', 'si'], dtype=object), 'sez2_costruzioneristrutturazione1': array(['<1919', '1992-2001', '1962-1971', '1919-1945', '1982-1991',
           '>2001', '1972-1981', '1946-1961', nan], dtype=object), 'sez7_morfologia_versante': array(['Pendio leggero', 'Pendio forte', 'Pianura', 'Cresta', nan],
          dtype=object), 'sez4_danno_strutturale_scale': array(['Danno Nullo', 'Danno D2-D3 Medio-Grave:<1/3',
           'Danno D2-D3 Medio-Grave:1/3-2/3', 'Danno D4-D5 Gravissimo:<1/3',
           'Danno D1 Leggero:<1/3',
           'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:1/3-2/3',
           'Danno D2-D3 Medio-Grave:>2/3', 'Danno D4-D5 Gravissimo:1/3-2/3',
           'Danno D1 Leggero:1/3-2/3', 'Danno D4-D5 Gravissimo:>2/3',
           'Danno D1 Leggero:>2/3',
           'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3',
           'Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:1/3-2/3',
           nan, 'Danno D2-D3 Medio-Grave:1/3-2/3, Danno D1 Leggero:<1/3',
           'Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:<1/3',
           'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:1/3-2/3',
           'Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:1/3-2/3',
           'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3',
           'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:>2/3',
           'Danno D4-D5 Gravissimo:>2/3, Danno D2-D3 Medio-Grave:<1/3',
           'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:>2/3',
           'Danno D4-D5 Gravissimo:>2/3, Danno D1 Leggero:<1/3',
           'Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:<1/3',
           'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3',
           'Danno D4-D5 Gravissimo:1/3-2/3, Danno D1 Leggero:1/3-2/3'],
          dtype=object)}
    


```python
#Count types of damage
dataframe[y_columns[0]].value_counts()
```




    Danno Nullo                                                                         54693
    Danno D1 Leggero:<1/3                                                                4025
    Danno D4-D5 Gravissimo:>2/3                                                          1549
    Danno D2-D3 Medio-Grave:<1/3                                                         1238
    Danno D2-D3 Medio-Grave:1/3-2/3                                                       954
    Danno D2-D3 Medio-Grave:>2/3                                                          776
    Danno D4-D5 Gravissimo:1/3-2/3                                                        698
    Danno D1 Leggero:1/3-2/3                                                              550
    Danno D4-D5 Gravissimo:<1/3                                                           543
    Danno D1 Leggero:>2/3                                                                 285
    Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3                                    44
    Danno D2-D3 Medio-Grave:1/3-2/3, Danno D1 Leggero:<1/3                                 26
    Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:1/3-2/3                           19
    Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:<1/3                           18
    Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:>2/3                                    16
    Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:1/3-2/3                                 16
    Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3                              15
    Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:>2/3                               8
    Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:1/3-2/3                         8
    Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:1/3-2/3                                   8
    Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:<1/3                                      8
    Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3        7
    Danno D4-D5 Gravissimo:>2/3, Danno D2-D3 Medio-Grave:<1/3                               4
    Danno D4-D5 Gravissimo:>2/3, Danno D1 Leggero:<1/3                                      3
    Danno D4-D5 Gravissimo:1/3-2/3, Danno D1 Leggero:1/3-2/3                                1
    Name: sez4_danno_strutturale_scale, dtype: int64



As we said, this data is categorical, so we need to convert it to numerical values. We will use the dictionary we created earlier to do this.


```python
# %% Various functions  to transform data into onehot encoding ##############################################################
def find_unique_positions(column, x, unique_values_dict):
    return  list(unique_values_dict[column]).index(x)
def int_to_onehot(column, x, unique_values_dict):
    onehot = [0]*(len(unique_values_dict[column]))
    onehot[x] = 1
    return onehot

def turn_to_numeric(dataframe, columns_exeptions = [], unique_values_dict = {}):
    for column in dataframe.columns:
        if column in columns_exeptions:
            continue
        print(column)
        dataframe[column] = dataframe[column].apply(lambda x: find_unique_positions(column=column, x=x, unique_values_dict=unique_values_dict))
    return dataframe
def numeric_to_onehot(dataframe, columns_exeptions = [], unique_values_dict = {}):
    for column in dataframe.columns:
        if column in columns_exeptions:
            continue
        dataframe[column] = dataframe[column].apply(lambda x: int_to_onehot(column=column, x=x, unique_values_dict=unique_values_dict))
    return dataframe
def onehot_to_signature(dataframe, Y_SIZE = Y_SIZE):
    signature_df = {'x':[], 'y':[]}
    for i in tqdm(range(len(dataframe))):
        signature_x = [onehot for onehot in dataframe.iloc[i,:-Y_SIZE]]
        signature_y = [onehot for onehot in dataframe.iloc[i,-Y_SIZE:]]
        signature_df['x'].append(flatten(signature_x))
        signature_df['y'].append(flatten(signature_y))
    return pd.DataFrame(signature_df)
```


```python
dataframe = turn_to_numeric(dataframe, columns_exeptions=coordinates_columns, unique_values_dict = unique_values_dict)
dataframe = numeric_to_onehot(dataframe, columns_exeptions=coordinates_columns, unique_values_dict = unique_values_dict)
dataframe = onehot_to_signature(dataframe)
```

    identificativoposizioneedificio
    sez3_struttura_orizzontale_1
    sez2_altezzamediapiano
    sez2_pianiinterrati
    sez3_struttura_verticale_1
    sez2_numeropiani
    sez2_superficiepiano
    sez3_pilastriisolati
    sez2_costruzioneristrutturazione1
    sez7_morfologia_versante
    sez4_danno_strutturale_scale
    

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 65523/65523 [00:46<00:00, 1411.97it/s]
    

We have now converted the data to numerical values. Let's take a look at it again.


```python
dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-0.8674214516922528, 0.5968959280707078, 1, 0...</td>
      <td>[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[-0.7894931324262333, 1.6023856802166563, 0, 1...</td>
      <td>[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[-0.664418864464821, 0.4405339408980041, 1, 0,...</td>
      <td>[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[-0.21363849853229364, 0.19104651928789274, 0,...</td>
      <td>[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[-0.21115149773491346, 0.1691897795630536, 0, ...</td>
      <td>[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
  </tbody>
</table>
</div>



This representation is much easier to work with, but is also very sparse. We will need to deal with this later.

In evaluation phase it will be useful to give a numerical score to each damage level. We do this by creating a dictionary that maps each damage level to a score. This is done 'by hand' following a simple rule. 


```python
damage_scores_dict = {'Danno Nullo':0,
                    'Danno D1 Leggero:<1/3':1,
                    'Danno D2-D3 Medio-Grave:<1/3':2,
                    'Danno D4-D5 Gravissimo:<1/3':4,
                    'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:1/3-2/3':8,
                    'Danno D4-D5 Gravissimo:1/3-2/3':8,
                    'Danno D2-D3 Medio-Grave:1/3-2/3':4,
                    'Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:<1/3':10,
                    'Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:<1/3':5,
                    'Danno D1 Leggero:1/3-2/3':2, 
                    'Danno D2-D3 Medio-Grave:>2/3':6,
                    'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3':6,
                    'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3':3,
                    'Danno D2-D3 Medio-Grave:1/3-2/3, Danno D1 Leggero:<1/3':5,
                    'Danno D4-D5 Gravissimo:>2/3':15, #rule exeption
                    'Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:1/3-2/3':12,
                    'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:>2/3':10,
                    'Danno D1 Leggero:>2/3':3,
                    'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:>2/3':5,
                    'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3':7,
                    'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:1/3-2/3':4,
                    'Danno D4-D5 Gravissimo:>2/3, Danno D2-D3 Medio-Grave:<1/3':14,
                    'Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:1/3-2/3':6,
                    'Danno D4-D5 Gravissimo:>2/3, Danno D1 Leggero:<1/3':13,
                    'Danno D4-D5 Gravissimo:1/3-2/3, Danno D1 Leggero:1/3-2/3':9,
                    'nan':0}
#sort dictionary by value
damage_scores_dict = {k: v for k, v in sorted(damage_scores_dict.items(), key=lambda item: item[1])}
damage_scores_dict
```




    {'Danno Nullo': 0,
     'nan': 0,
     'Danno D1 Leggero:<1/3': 1,
     'Danno D2-D3 Medio-Grave:<1/3': 2,
     'Danno D1 Leggero:1/3-2/3': 2,
     'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3': 3,
     'Danno D1 Leggero:>2/3': 3,
     'Danno D4-D5 Gravissimo:<1/3': 4,
     'Danno D2-D3 Medio-Grave:1/3-2/3': 4,
     'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:1/3-2/3': 4,
     'Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:<1/3': 5,
     'Danno D2-D3 Medio-Grave:1/3-2/3, Danno D1 Leggero:<1/3': 5,
     'Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:>2/3': 5,
     'Danno D2-D3 Medio-Grave:>2/3': 6,
     'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3': 6,
     'Danno D4-D5 Gravissimo:<1/3, Danno D1 Leggero:1/3-2/3': 6,
     'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:<1/3, Danno D1 Leggero:<1/3': 7,
     'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:1/3-2/3': 8,
     'Danno D4-D5 Gravissimo:1/3-2/3': 8,
     'Danno D4-D5 Gravissimo:1/3-2/3, Danno D1 Leggero:1/3-2/3': 9,
     'Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:<1/3': 10,
     'Danno D4-D5 Gravissimo:<1/3, Danno D2-D3 Medio-Grave:>2/3': 10,
     'Danno D4-D5 Gravissimo:1/3-2/3, Danno D2-D3 Medio-Grave:1/3-2/3': 12,
     'Danno D4-D5 Gravissimo:>2/3, Danno D1 Leggero:<1/3': 13,
     'Danno D4-D5 Gravissimo:>2/3, Danno D2-D3 Medio-Grave:<1/3': 14,
     'Danno D4-D5 Gravissimo:>2/3': 15}




```python

```
