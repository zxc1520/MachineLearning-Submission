# LAPORAN PRAKTIKUM TUGAS MINGGU KE-3

---

## Nama : Afif Qomarul Ghulam

## NIM : 2041720176

## Nomor Absen : 1

---

## Pemanggilan Librari

```py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
```

pemanggilan library python ini ditujukan agar dapat mengakses fungsi statistika untuk proses pelatihan data

## Pemanggilan Data

```py
df = pd.read_csv('50_Startups.csv')
df.head()
```

pemanggilan datasheet ini yang akan diambil datanya untuk di-train

## Pembuatan Tabel OneHotEncoder

Pertama didefinisikan terlebih dahulu pada baris mana letak fitur kategorinya

```py
transformer_list = [
     ('encoded', OneHotEncoder(dtype='int'), [3]),
     ('skip', 'passthrough', ["R&D Spend", "Administration", "Marketing Spend", "Profit"])
]
```

kemudian dilakukan proses OneHotEncoder

```py
ct = ColumnTransformer(transformer_list)

tr = ct.fit_transform(df)
```

jika sudah selesai, maka dilakukan pengecekan apakah fitur tersebut sudah berhasil atau tidak, dengan menampilkan keseluruhan tabel

```py
df2 = pd.DataFrame(tr, columns=ct.get_feature_names())
df2.head()
```

## Penyeleksian Data

---

Tahap selanjutnya ialah menyeleksi data dari awal dengan skrip berikut

```py
X = df2.iloc[:, :-1].values
y = df2.iloc[:, -1].values
```

kemudian dilakukan reshape untuk merubah data yang sebelumnya berupa 1D array menjadi 2D array, dikarenakan library ini membutuhkan 2D Array.

```py
y = y.reshape(len(y), 1)
y.shape
```

## Pemisahan data

---

Dalam machine learning sering sekali data dipisah sebelum dilakukan proses training

```py
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=50)
```

dalam skrip tersebut data `X_train`, `X_test`, `y_train`, dan `y_test` dipisah kemudian diberikan parameter `test_size` berupa bilangan float `0.2` dan `random_state` berfungsi untuk melakukan pengacakan terhadap data sebelum dipisah

## Proses Train Multiple Data

---

```py
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)
```

## Penggabungan Data

---

```py
import numpy as np

con = np.concatenate((y_test, y_pred), axis=1)
con
```
