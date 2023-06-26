from flask import Flask, render_template, request
import pandas as pd
from pyECLAT import ECLAT
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Baca Data Transaksi dan Data Barang
dataTransaksi = pd.read_csv("Data/transaksi.csv")
dataBarang = pd.read_csv("Data/dataBarang.csv")

# Data ECLAT
dataEclat = pd.read_csv("Data/transaksiECLAT2.csv", header=None)

# Data KNN
dataKNN = pd.read_csv("DATA/AbdiTraining.csv", sep=";")

# Implementasi KNN
def implementasiKNN(k):
    listAkurasi = []
    prediksi = []
    prediksi.append(dataKNN['Id barang'])
    listPrediksi = []
    listDataX = [['H1','H2','H3'],['H4','H5','H6'],['H7','H8','H9'],['H10','H11','H12']]
    listDataY = ['H4','H7','H10','H13']

    knn = KNeighborsClassifier(n_neighbors=k)
    for i in range(len(listDataY)):
        data_X = dataKNN[listDataX[i]]
        data_Y = dataKNN[listDataY[i]]
        knn.fit(data_X, data_Y)
        prediksiKNN = knn.predict(data_X)
        akurasi = accuracy_score(data_Y, prediksiKNN)
        prediksi.append(data_Y)
        prediksi.append(prediksiKNN)
        listAkurasi.append(round((akurasi*100),2))
    for i in range(len(prediksi[0])):
        tempPrediksi = [prediksi[0][i], prediksi[1][i], prediksi[2][i], prediksi[3][i],
        prediksi[4][i], prediksi[5][i], prediksi[6][i], prediksi[7][i], prediksi[8][i]]
        listPrediksi.append(tempPrediksi)

    return [listPrediksi, listAkurasi]

# Implementasi ECLAT
def ImplemenECLAT(support, kombinasiMin, kombinasiMax):
    my_eclat = ECLAT(data=dataEclat, verbose=True)
    rule_indices, rule_supports = my_eclat.fit(min_support=support, min_combination=kombinasiMin, 
                                    max_combination=kombinasiMax)
    hasil = []
    for i in range(len(rule_supports)):
        aturan = list(rule_supports.items())[i][0]
        supportPersen = list(rule_supports.items())[i][1]
        supportCount = len(list(rule_indices.items())[i][1])
        temp = [aturan, supportPersen, supportCount]
        hasil.append(temp)
    return hasil

app = Flask(__name__)

@app.route("/")
def index():
    judul = "ABDI UMKM"
    return render_template("index.html", judul=judul)

@app.route("/tampilDataTransaksi")
def tampilDataTransaksi():
    judul = "Data Transaksi"
    return render_template("tampilDataTransaksi.html", column_names=dataTransaksi.columns.values, 
                            row_data=list(dataTransaksi.values.tolist()), zip=zip, judul=judul)

@app.route("/tampilDataBarang")
def tampilDataBarang():
    judul = "Data Barang"
    return render_template("tampilDataBarang.html", column_names=dataBarang.columns.values, 
                            row_data=list(dataBarang.values.tolist()), zip=zip, judul=judul)

@app.route("/eclat")
def eclat():
    judul = "ECLAT"
    return render_template("eclat.html", judul=judul)

@app.route("/eclatProcess", methods=["POST"])
def eclatProcess():
    judul = "ECLAT Process"
    support = round(float(request.form['support'])/100,2)
    MinKombinasi = int(request.form['MinKombinasi'])
    MaxKombinasi = int(request.form['MaxKombinasi'])
    hasil = ImplemenECLAT(support, MinKombinasi, MaxKombinasi)
    return render_template("eclatProcess.html", supportPersen=support, MinKombinasi_=MinKombinasi, 
                            MaxKombinasi_=MaxKombinasi, hasil_=hasil, judul=judul)

@app.route("/knn")
def knn():
    judul = "KNN"
    return render_template("knn.html", judul=judul)

@app.route("/knnProcess", methods=["POST"])
def knnProcess():
    judul = "KNN Process"
    k = int(request.form['k'])
    hasil = implementasiKNN(k)
    prediksi = hasil[0]
    akurasi = hasil[1]
    return render_template("knnProcess.html", prediksi=prediksi, akurasi=akurasi, k=k, judul=judul)

if __name__ == "__main__":
    app.run(debug=True)