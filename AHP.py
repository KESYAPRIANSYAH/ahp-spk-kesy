import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Dictionary to store responses from multiple users
user_responses = defaultdict(list)

@st.cache_data
def get_weight(A, str_label, labels, user_id):
    n = A.shape[0]
    e_vals, e_vecs = np.linalg.eig(A)
    lamb = np.max(np.real(e_vals))
    w = np.real(e_vecs[:, np.argmax(np.real(e_vals))])
    w = w / np.sum(w)  # Normalisasi
    
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24,
          7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
    ci = (lamb - n) / (n - 1)
    cr = ci / ri.get(n, float('inf'))
    
    st.write(f"### Vektor Eigen yang Dinormalisasi untuk {str_label} (User {user_id}):")
    df_weight = pd.DataFrame(w, columns=['Bobot'], index=labels)
    st.table(df_weight)
    
    st.write('CR = %f' % cr)
    if cr > 0.1:
        st.error(f"⚠️ Gagal pemeriksaan konsistensi pada {str_label} (User {user_id})")

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
def calculate_ahp(A, B, n, m, criterias, alternatives, user_id):
    for i in range(n):
        for j in range(i, n):
            if i != j:
                A[j][i] = float(1 / A[i][j])
    dfA = pd.DataFrame(A, index=criterias, columns=criterias)
    st.markdown(f" #### Tabel Kriteria (User {user_id})")
    st.table(dfA)

    for k in range(n):
        for i in range(m):
            for j in range(i, m):
                if i != j:
                    B[k][j][i] = float(1 / B[k][i][j])
    st.write("---")

    for i in range(n):
        dfB = pd.DataFrame(B[i], index=alternatives, columns=alternatives)
        st.markdown(f" #### Tabel Alternatif untuk Kriteria {criterias[i]} (User {user_id})")
        st.table(dfB)

    W2 = get_weight(A, "Tabel Kriteria", criterias, user_id)
    W3 = np.zeros((n, m))

    for i in range(n):
        w3 = get_weight(B[i], f"Tabel Alternatif untuk Kriteria {criterias[i]}", alternatives, user_id)
        W3[i] = w3

    W = np.dot(W2, W3)
    
    df_result = pd.DataFrame({'Alternatif': alternatives, 'Skor Akhir': W})
    df_result = df_result.sort_values('Skor Akhir', ascending=False).reset_index(drop=True)
    df_result['Ranking'] = df_result['Skor Akhir'].rank(ascending=False).astype(int)

    # Plot grafik hasil AHP
    st.pyplot(plot_graph(W2, criterias, "Kriteria", f"Bobot Kriteria (User {user_id})"))
    st.pyplot(plot_graph(W, alternatives, "Alternatif", f"Alternatif Optimal untuk Kriteria yang Diberikan (User {user_id})"))
    st.balloons()

    # Menyimpan hasil perhitungan AHP untuk user saat ini
    user_responses[user_id].append({
        "comparison_matrix_A": A,
        "comparison_matrix_B": B,
        "priority_vector_A": W2,
        "priority_vector_B": W3,
        "final_score": W,
        "ranking": df_result[['Alternatif', 'Skor Akhir', 'Ranking']]
    })

    # Menampilkan Hasil Akhir dengan Ranking di bagian paling bawah
    st.write(f"### Hasil Akhir AHP dengan Ranking (User {user_id}):")
    st.table(df_result[['Alternatif', 'Skor Akhir', 'Ranking']])

def delete_user_response(user_id, index):
    """
    Delete a specific response for the given user.
    """
    del user_responses[user_id][index]

def main():
    st.set_page_config(page_title="Kalkulator AHP ", page_icon=":bar_chart:")
    st.header("Kalkulator AHP Untuk Menentukan Jenis Gamifikasi Pop-Up Campaign")
    st.sidebar.title(" Kriteria & Alternatif")

    # Petunjuk Pengisian AHP
    st.sidebar.info("""
    ### Petunjuk Pengisian AHP
    
    Untuk mendapatkan hasil yang optimal dan konsisten, harap perhatikan langkah-langkah berikut saat mengisi nilai perbandingan:
    
    1. Masukkan Input Metrik dan Nama Jenis Gamifikasi dengan tanda , misal CTR, CR , IMPRESSION 
    2. **Konsistensi**: Jika Kriteria A lebih penting dari Kriteria B, dan Kriteria B lebih penting dari Kriteria C, maka Kriteria A seharusnya jauh lebih penting daripada Kriteria C.
    
    3. **Skala Pengisian**: Gunakan skala **1 hingga 9**:
       - 1: Sama penting
       - 3: Sedikit lebih penting
       - 5: Lebih penting
       - 7: Sangat lebih penting
       - 9: Mutlak lebih penting
    
    4. **Perbandingan Simetris**: Jika Anda menilai Kriteria A lebih penting daripada Kriteria B, maka sebaliknya, nilai Kriteria B terhadap Kriteria A harus otomatis terbalik.
       ### Penggunaan Nilai 2, 4, 6, dan 8:
    - **Nilai 2**: Kriteria A sedikit lebih penting dari Kriteria B.
    - **Nilai 4**: Kriteria A lebih penting dari Kriteria B, tetapi tidak terlalu jauh.
    - **Nilai 6**: Kriteria A cukup lebih penting dari Kriteria B.
    - **Nilai 8**: Kriteria A sangat lebih penting dibandingkan Kriteria B.                 
    """)
    
    cri = st.sidebar.text_input("Masukkan Kriteria Metrik")
    alt = st.sidebar.text_input("Masukkan Alternatif Jenis Gamifikasi")
    criterias = cri.split(",")
    alternatives = alt.split(",")

    if cri and alt:
        with st.expander("Bobot Kriteria"):
            st.subheader("Perbandingan Berpasangan untuk Kriteria")
            n = len(criterias)
            A = np.zeros((n, n))

            user_ids = st.multiselect("Pilih User", [], format_func=lambda x: f"User {x}")

            for user_id in user_ids:
                st.write(f"## Perbandingan Berpasangan untuk User {user_id}")
                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            A[i][j] = 1
                        else:
                            st.markdown(f" ##### Kriteria {criterias[i]} dibandingkan dengan Kriteria {criterias[j]}")
                            criteriaradio = st.radio(f"Pilih kriteria yang lebih prioritas untuk User {user_id}", (criterias[i], criterias[j]), horizontal=True)

                            if criteriaradio == criterias[i]:
                                A[i][j] = st.slider(f"Seberapa jauh {criterias[i]} lebih penting dibandingkan {criterias[j]} ?", 1, 9, 1)
                                A[j][i] = float(1/A[i][j])
                            else:
                                A[j][i] = st.slider(f"Seberapa jauh {criterias[j]} lebih penting dibandingkan {criterias[i]} ?", 1, 9, 1)
                                A[i][j] = float(1/A[j][i])
                calculate_ahp(A, B, n, m, criterias, alternatives, user_id)

        with st.expander("Bobot Alternatif"):
            st.subheader("Perbandingan Berpasangan untuk Alternatif")
            m = len(alternatives)
            B = np.zeros((n, m, m))

            for user_id in user_ids:
                st.write(f"## Perbandingan Berpasangan untuk User {user_id}")
                for k in range(n):
                    st.write("---")
                    st.markdown(f" ##### Perbandingan Alternatif untuk Kriteria {criterias[k]}")

                    for i in range(m):
                        for j in range(i, m):
                            if i == j:
                                B[k][i][j] = 1
                            else:
                                alternativeradio = st.radio(f"Pilih alternatif yang lebih prioritas untuk kriteria {criterias[k]} (User {user_id})", (alternatives[i], alternatives[j]), horizontal=True)

                                if alternativeradio == alternatives[i]:
                                    B[k][i][j] = st.slider(f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[i]} lebih baik dibandingkan {alternatives[j]} ?", 1, 9, 1)
                                    B[k][j][i] = float(1/B[k][i][j])
                                else:
                                    B[k][j][i] = st.slider(f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[j]} lebih baik dibandingkan {alternatives[i]} ?", 1, 9, 1)
                                    B[k][i][j] = float(1/B[k][j][i])
                calculate_ahp(A, B, n, m, criterias, alternatives, user_id)

        btn = st.button("Hitung AHP")
        st.write("##")

        if btn:
            for user_id in user_ids:
                calculate_ahp(A, B, n, m, criterias, alternatives, user_id)

        # Menampilkan hasil perhitungan AHP untuk semua user
        with st.expander("Hasil Perhitungan AHP untuk Semua User"):
            if user_responses:
                for user_id, responses in user_responses.items():
                    st.write(f"## Hasil AHP untuk User {user_id}")
                    for i, response in enumerate(responses):
                        st.write(f"### Perhitungan AHP {i+1}")
                        st.write("Matriks Perbandingan Kriteria:")
                        st.table(pd.DataFrame(response["comparison_matrix_A"], index=criterias, columns=criterias))
                        st.write("Bobot Kriteria:")
                        st.table(pd.DataFrame(response["priority_vector_A"], index=criterias, columns=["Bobot"]))
                        st.write("Matriks Perbandingan Alternatif:")
                        for j in range(n):
                            st.write(f"Matriks Perbandingan Alternatif untuk Kriteria {criterias[j]}:")
                            st.table(pd.DataFrame(response["comparison_matrix_B"][j], index=alternatives, columns=alternatives))
                        st.write("Bobot Alternatif:")
                        st.table(pd.DataFrame(response["priority_vector_B"], index=criterias, columns=alternatives))
                        st.write("Skor Akhir dan Ranking:")
                        st.table(response["ranking"])
                        st.button(f"Hapus Hasil Perhitungan {i+1} (User {user_id})", key=f"delete_button_{user_id}_{i}", on_click=delete_user_response, args=(user_id, i))
            else:
                st.write("Belum ada hasil AHP yang disimpan.")

if __name__ == '__main__':
    main()
