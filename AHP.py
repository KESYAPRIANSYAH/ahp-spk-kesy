import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def get_weight(A, str_label):
    n = A.shape[0]
    e_vals, e_vecs = np.linalg.eig(A)
    lamb = np.max(np.real(e_vals))
    w = np.real(e_vecs[:, np.argmax(np.real(e_vals))])
    w = w / np.sum(w)  # Normalisasi
    
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24,
          7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
    ci = (lamb - n) / (n - 1)
    cr = ci / ri.get(n, float('inf'))
    
    st.write(f"### Vektor Eigen yang Dinormalisasi untuk {str_label}:")
    df_weight = pd.DataFrame(w, columns=['Bobot'])
    df_weight['Ranking'] = df_weight['Bobot'].rank(ascending=False, method='min')
    st.table(df_weight)
    
    st.write('CR = %f' % cr)
    if cr > 0.1:
        st.error(f"⚠️ Gagal pemeriksaan konsistensi pada {str_label}")

    return w


def plot_graph(x, y, ylabel, title):
    fig, ax = plt.subplots()
    ax.bar(y, x, color='#088eff')
    ax.set_facecolor('#F0F2F6')
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Nilai")
    return fig


@st.cache_data
def calculate_ahp(A, B, n, m, criterias, alternatives):
    for i in range(n):
        for j in range(i, n):
            if i != j:
                A[j][i] = float(1 / A[i][j])
    dfA = pd.DataFrame(A, index=criterias, columns=criterias)
    st.markdown(" #### Tabel Kriteria")
    st.table(dfA)

    for k in range(n):
        for i in range(m):
            for j in range(i, m):
                if i != j:
                    B[k][j][i] = float(1 / B[k][i][j])
    st.write("---")

    for i in range(n):
        dfB = pd.DataFrame(B[i], index=alternatives, columns=alternatives)
        st.markdown(f" #### Tabel Alternatif untuk Kriteria {criterias[i]}")
        st.table(dfB)

    W2 = get_weight(A, "Tabel Kriteria")
    W3 = np.zeros((n, m))

    for i in range(n):
        w3 = get_weight(B[i], f"Tabel Alternatif untuk Kriteria {criterias[i]}")
        W3[i] = w3

    W = np.dot(W2, W3)
    
    df_result = pd.DataFrame({'Alternatif': alternatives, 'Skor Akhir': W})
    df_result['Ranking'] = df_result['Skor Akhir'].rank(ascending=False, method='min')
    st.write("### Hasil Akhir AHP:")
    st.table(df_result.sort_values('Ranking'))

    st.pyplot(plot_graph(W2, criterias, "Kriteria", "Bobot Kriteria"))
    st.pyplot(plot_graph(W, alternatives, "Alternatif", "Alternatif Optimal untuk Kriteria yang Diberikan"))
    st.balloons()


def main():
    st.set_page_config(page_title="Kalkulator AHP", page_icon=":bar_chart:")
    st.header("Kalkulator AHP")
    st.sidebar.title("Kriteria & Alternatif")

    st.sidebar.info("""### Petunjuk Pengisian AHP...""")
    
    cri = st.sidebar.text_input("Masukkan Kriteria")
    alt = st.sidebar.text_input("Masukkan Alternatif")
    criterias = cri.split(",")
    alternatives = alt.split(",")

    if cri and alt:
        with st.expander("Bobot Kriteria"):
            st.subheader("Perbandingan Berpasangan untuk Kriteria")
            n = len(criterias)
            A = np.zeros((n, n))

            for i in range(n):
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

            for k in range(n):
                st.write("---")
                st.markdown(f" ##### Perbandingan Alternatif untuk Kriteria {criterias[k]}")

                for i in range(m):
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
