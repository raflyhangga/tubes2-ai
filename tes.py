import pandas as pd

def change_name(name):
    return name.upper()


# Membuat DataFrame
data = {
    "Nama": ["Ali", "Budi", "Citra"],
    "Usia": [23, 27, 25],
    "Kota": ["Jakarta", "Bandung", "Surabaya"],
    "Gaji": [10000000, 15000000, 12000000],
    "Keterangan": ["Karyawan", "Manajer", "Karyawan"]
}

df = pd.DataFrame(data)
# print(df.head())
print(df["Nama"])
df2 = df[["Nama", "Usia", "Kota"]]
# df2 = df["Nama", "Usia", "Kota"].apply(change_name)
print(df)
print(df2)