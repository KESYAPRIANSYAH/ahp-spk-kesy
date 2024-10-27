import streamlit as st  # Untuk membangun dan berbagi aplikasi web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Untuk perbandingan dan analisis grafis


@st.cache_data
def get_weight(A, str_label):
    n = A.shape[0]
    # Menghitung eigen value dan eigen vector
    e_vals, e_vecs = np.linalg.eig(A)
    # Mengambil eigen value terbesar (bagian real)
    lamb = np.max(np.real(e_vals))
    # Mengambil eigen vector yang sesuai dengan eigen value terbesar (bagian real)
    w = np.real(e_vecs[:, np.argmax(np.real(e_vals))])
    w = w / np.sum(w)  # Normalisasi

    # Pemeriksaan Konsistensi
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24,
          7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
    ci = (lamb - n) / (n - 1)  # Menghitung Consistency Index (CI)
    cr = ci / ri.get(n, float('inf'))  # Menghitung Consistency Ratio (CR)

    # Output hasil
    st.write("Vektor eigen yang telah dinormalisasi (Priority Vector):")
    st.write(np.round(w, 4))  # Menampilkan dengan pembulatan
    st.write('CR = %.4f' % cr)  # Menampilkan CR dengan pembulatan
    if cr > 0.1:
        st.error(f"⚠️ Gagal pemeriksaan konsistensi pada {str_label}. Periksa kembali input perbandingan.")

    return w


def plot_graph(x, y, ylabel, title):
    # Membuat diagram batang horizontal
    fig, ax = plt.subplots()
    ax.barh(y, x, color='#088eff')  # Perbaikan bar chart menjadi horizontal
    ax.set_facecolor('#F0F2F6')
    # Set judul dan label sumbu
    ax.set_title(title)
    ax.set_xlabel("Nilai")
    ax.set_ylabel(ylabel)
    return fig


@st.cache_data
def calculate_ahp(A, B, n, m, criterias, alternatives):
    # Normalisasi matriks perbandingan berpasangan untuk kriteria
    for i in range(0, n):
        for j in range(i, n):
            if i != j:
                A[j][i] = float(1/A[i][j])
    # Cetak Tabel Kriteria
    dfA = pd.DataFrame(A, index=criterias, columns=criterias)
    st.markdown(" #### Tabel Kriteria")
    st.table(dfA)

    # Normalisasi matriks perbandingan berpasangan untuk alternatif sesuai kriteria
    for k in range(0, n):
        for i in range(0, m):
            for j in range(i, m):
                if i != j:
                    B[k][j][i] = float(1/B[k][i][j])
    st.write("---")

    # Cetak Tabel Alternatif sesuai Kriteria
    for i in range(0, n):
        dfB = pd.DataFrame(B[i], index=alternatives, columns=alternatives)
        st.markdown(f" #### Tabel Alternatif untuk Kriteria {criterias[i]}")
        st.table(dfB)

    # Menghitung bobot kriteria
    W2 = get_weight(A, "Tabel Kriteria")
    W3 = np.zeros((n, m))

    # Menghitung bobot alternatif untuk setiap kriteria
    for i in range(0, n):
        w3 = get_weight(B[i], f"Tabel Alternatif untuk Kriteria {criterias[i]}")
        W3[i] = w3

    # Menghitung skor akhir untuk alternatif
    W = np.dot(W2, W3)

    # Plot grafis hasil AHP
    st.pyplot(plot_graph(W2, criterias, "Kriteria", "Bobot Kriteria"))
    st.pyplot(plot_graph(W, alternatives, "Alternatif", "Alternatif Optimal untuk Kriteria yang Diberikan"))
    st.balloons()


def main():
    st.set_page_config(page_title="Kalkulator AHP", page_icon=":bar_chart:")
    st.header("Kalkulator AHP")
    st.sidebar.title("Kriteria & Alternatif")

    # Instruksi Pengguna di Sidebar
    st.sidebar.info("""
    ### Petunjuk Pengisian AHP
    
    1. **Konsistensi**: Harus konsisten antara perbandingan.
    2. **Skala Pengisian**: Gunakan skala **1 hingga 9**:
       - 1: Sama penting
       - 3: Sedikit lebih penting
       - 5: Lebih penting
       - 7: Sangat lebih penting
       - 9: Mutlak lebih penting
    """)
    

    cri = st.sidebar.text_input("Masukkan Kriteria")
    alt = st.sidebar.text_input("Masukkan Alternatif")
    criterias = cri.split(",")
    alternatives = alt.split(",")

    if cri and alt:
        with st.expander("Bobot Kriteria"):
            st.subheader("Perbandingan Berpasangan untuk Kriteria")
            st.write("---")
            n = len(criterias)
            A = np.zeros((n, n))

            # Input perbandingan berpasangan untuk kriteria
            for i in range(0, n):
                for j in range(i, n):
                    if i == j:
                        A[i][j] = 1
                    else:
                        st.markdown(f" ##### Kriteria {criterias[i]} dibandingkan dengan Kriteria {criterias[j]}")
                        criteriaradio = st.radio("Pilih kriteria yang lebih prioritas ", (criterias[i], criterias[j]), horizontal=True)

                        if criteriaradio == criterias[i]:
                            A[i][j] = st.slider(f"Seberapa jauh {criterias[i]} lebih penting dibandingkan {criterias[j]} ?", 1, 9, 1)
                            A[j][i] = float(1/A[i][j])
                        else:
                            A[j][i] = st.slider(f"Seberapa jauh {criterias[j]} lebih penting dibandingkan {criterias[i]} ?", 1, 9, 1)
                            A[i][j] = float(1/A[j][i])

        with st.expander("Bobot Alternatif"):
            st.subheader("Perbandingan Berpasangan untuk Alternatif")
            m = len(alternatives)
            B = np.zeros((n, m, m))

            # Input perbandingan berpasangan untuk alternatif berdasarkan setiap kriteria
            for k in range(0, n):
                st.write("---")
                st.markdown(f" ##### Perbandingan Alternatif untuk Kriteria {criterias[k]}")

                for i in range(0, m):
                    for j in range(i, m):
                        if i == j:
                            B[k][i][j] = 1
                        else:
                            alternativeradio = st.radio(f"Pilih alternatif yang lebih prioritas untuk kriteria {criterias[k]}", (alternatives[i], alternatives[j]), horizontal=True)

                            if alternativeradio == alternatives[i]:
                                B[k][i][j] = st.slider(f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[i]} lebih baik dibandingkan {alternatives[j]} ?", 1, 9, 1)
                                B[k][j][i] = float(1/B[k][i][j])
                            else:
                                B[k][j][i] = st.slider(f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[j]} lebih baik dibandingkan {alternatives[i]} ?", 1, 9, 1)
                                B[k][i][j] = float(1/B[k][j][i])

        btn = st.button("Hitung AHP")
        st.write("##")

        if btn:
            calculate_ahp(A, B, n, m, criterias, alternatives)


if __name__ == '__main__':
    main()
