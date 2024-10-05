import streamlit as st  #Untuk membuat dan membagikan aplikasi web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #Untuk perbandingan dan analisis grafis


@st.cache_data
def get_weight(A, str):
    n = A.shape[0]
    e_vals, e_vecs = np.linalg.eig(A)
    lamb = max(e_vals)
    w = e_vecs[:, e_vals.argmax()]
    w = w / np.sum(w)  # Normalisasi
    # Pengecekan Konsistensi
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24,
          7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
    ci = (lamb - n) / (n - 1)
    cr = ci / ri[n]
    print("Vektor eigen yang dinormalisasi:")
    print(w)
    print('CR = %f' % cr)
    if cr >= 0.1:
        print("Gagal dalam pengecekan konsistensi "+str)
        st.error("Gagal dalam pengecekan konsistensi "+str)

    return w


def plot_graph(x, y, ylabel, title):
    # Membuat grafik batang horizontal
    fig, ax = plt.subplots()
    ax.bar(y, x, color='#03efff')
    ax.set_facecolor('#F0F2F6')
    # Atur judul dan label sumbu
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Nilai")
    return fig


@st.cache_data
def calculate_ahp(A, B, n, m, criterias, alternatives):

    for i in range(0, n):
        for j in range(i, n):
            if i != j:
                A[j][i] = float(1/A[i][j])
    dfA = pd.DataFrame(A)
    st.markdown(" #### Tabel Kriteria")
    st.table(dfA)
    for k in range(0, n):
        for i in range(0, m):
            for j in range(i, m):
                if i != j:
                    B[k][j][i] = float(1/B[k][i][j])
    st.write("---")
    for i in range(0, n):
        dfB = pd.DataFrame(B[i])
        st.markdown(" #### Tabel Alternatif untuk Kriteria " + criterias[i])
        st.table(dfB)
    W2 = get_weight(A, "Tabel Kriteria")
    W3 = np.zeros((n, m))
    for i in range(0, n):
        w3 = get_weight(B[i], "Tabel Alternatif untuk Kriteria "+ criterias[i])
        W3[i] = w3
    W = np.dot(W2, W3)

    st.pyplot(plot_graph(W2, criterias, "Kriteria", "Bobot Kriteria"))

    st.pyplot(plot_graph(W, alternatives, "Alternatif", "Alternatif Terbaik untuk Kriteria yang Diberikan"))
    st.balloons()


def main():
    st.set_page_config(page_title="Kalkulator AHP", page_icon=":bar_chart:")
    st.header("Kalkulator AHP")

    st.sidebar.title("Kriteria & Alternatif")
    cri = st.sidebar.text_input("Masukkan Kriteria")
    alt = st.sidebar.text_input("Masukkan Alternatif")
    criterias = cri.split(",")
    alternatives = alt.split(",")
    st.sidebar.info("Masukkan beberapa nilai Kriteria & Alternatif, dipisahkan oleh koma tanpa spasi.")
    st.sidebar.info("Contoh: Mobil,Bus,Truk")

    if cri and alt:
        with st.expander("Bobot Kriteria"):
            st.subheader("Perbandingan Berpasangan untuk Kriteria")
            st.write("---")
            n = len(criterias)
            A = np.zeros((n, n))
            Aradio = np.zeros((n,n))
            for i in range(0, n):
                for j in range(i, n):
                    if i == j:
                        A[i][j] = 1
                    else:
                        st.markdown(" ##### Perbandingan Kriteria "+criterias[i] + " dengan Kriteria " +criterias[j])
                        criteriaradio = st.radio("Pilih kriteria yang lebih penting ",(criterias[i], criterias[j],), horizontal=True)
                        if criteriaradio == criterias[i]:
                            A[i][j] = st.slider(label="Seberapa lebih penting "+ criterias[i] +" dibandingkan dengan "+criterias[j]+ " ?",min_value= 1,max_value= 9,value= 1)
                            A[j][i] = float(1/A[i][j])
                        else:
                            A[j][i] = st.slider(label="Seberapa lebih penting "+ criterias[j] +" dibandingkan dengan "+criterias[i]+ " ?",min_value= 1,max_value= 9,value= 1)
                            A[i][j] = float(1/A[j][i])
                
        with st.expander("Bobot Alternatif"):
            st.subheader("Perbandingan Berpasangan untuk Alternatif")
            m = len(alternatives)
            B = np.zeros((n, m, m))

            for k in range(0, n):
                st.write("---")
                st.markdown(" ##### Perbandingan Alternatif untuk Kriteria "+criterias[k])
                for i in range(0, m):
                    for j in range(i, m):
                        if i == j:
                            B[k][i][j] = 1
                        else:
                            alternativeradio = st.radio("Pilih alternatif yang lebih penting untuk kriteria "+criterias[k] ,(alternatives[i], alternatives[j],), horizontal=True)
                            if alternativeradio == alternatives[i]:
                                B[k][i][j] = st.slider("Berdasarkan Kriteria " + criterias[k] + ", seberapa lebih penting " + alternatives[i] + " dibandingkan dengan " + alternatives[j]+" ?", 1, 9, 1)
                                B[k][j][i] = float(1/B[k][i][j])
                            else:
                                B[k][j][i] = st.slider("Berdasarkan Kriteria " + criterias[k] + ", seberapa lebih penting " + alternatives[j] + " dibandingkan dengan " + alternatives[i]+" ?", 1, 9, 1)
                                B[k][i][j] = float(1/B[k][j][i])

                
        btn = st.button("Hitung AHP")
        st.write("##")
        if btn:
            calculate_ahp(A, B, n, m, criterias, alternatives)


if __name__ == '__main__':
    main()
