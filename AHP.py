import streamlit as st
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Initialize session state for storing responses and calculation status
if 'responses' not in st.session_state:
    st.session_state.responses = []

if 'calculated' not in st.session_state:
    st.session_state.calculated = False

# Save response to session state
def save_response(respondent_name, A, B, W, criterias, alternatives):
    response_data = {
        'respondent_name': str(respondent_name),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'criteria_data': A.tolist(),
        'alternative_data': B.tolist(),
        'final_scores': [float(score) for score in W],
        'criterias': list(criterias),
        'alternatives': list(alternatives)
    }
    st.session_state.responses.append(response_data)
    st.session_state.calculated = True  # Set calculated to True

# Function to calculate AHP scores
def calculate_ahp(A, B, n, m, criterias, alternatives):
    # Normalize and calculate weights for the criteria matrix
    e_vals, e_vecs = np.linalg.eig(A)
    lamb = np.max(np.real(e_vals))
    W2 = np.real(e_vecs[:, np.argmax(np.real(e_vals))])
    W2 = W2 / np.sum(W2)  # Normalize weights

    # Normalize and calculate weights for each alternatives matrix
    W3 = np.zeros((n, m))
    for k in range(n):
        e_vals, e_vecs = np.linalg.eig(B[k])
        W = np.real(e_vecs[:, np.argmax(np.real(e_vals))])
        W3[k] = W / np.sum(W)  # Normalize weights

    # Calculate final scores
    W = np.dot(W2, W3)
    return W, W2, W3

def main():
    st.set_page_config(page_title="Kalkulator AHP", page_icon=":bar_chart:")
    st.header("Kalkulator AHP Untuk Menentukan Jenis Gamifikasi Pop-Up Campaign")

    # Sidebar with instructions
    st.sidebar.title("Kriteria & Alternatif")
    st.sidebar.info("""
    ### Petunjuk Pengisian AHP
    Untuk mendapatkan hasil yang optimal dan konsisten, harap perhatikan langkah-langkah berikut saat mengisi nilai perbandingan:

    1. Masukkan Input Metrik dan Nama Jenis Gamifikasi dengan tanda koma, misal CTR, CR, IMPRESSION.
    2. **Konsistensi**: Jika Kriteria A lebih penting dari Kriteria B, dan Kriteria B lebih penting dari Kriteria C, maka Kriteria A seharusnya jauh lebih penting daripada Kriteria C.
    3. **Skala Pengisian**: Gunakan skala **1 hingga 9**:
       - 1: Sama penting
       - 3: Sedikit lebih penting
       - 5: Lebih penting
       - 7: Sangat lebih penting
       - 9: Mutlak lebih penting
    4. **Perbandingan Simetris**: Jika Anda menilai Kriteria A lebih penting daripada Kriteria B, maka sebaliknya, nilai Kriteria B terhadap Kriteria A harus otomatis terbalik.
    """)

    # Add tabs for input and analysis
    tab1, tab2 = st.tabs(["Input Data", "Analisis Responden"])

    with tab1:
        respondent_name = st.text_input("Nama Responden")
        cri = st.text_input("Masukkan Kriteria Metrik (pisahkan dengan koma)")
        alt = st.text_input("Masukkan Alternatif Jenis Gamifikasi (pisahkan dengan koma)")

        if cri and alt:
            criterias = [c.strip() for c in cri.split(",")]
            alternatives = [a.strip() for a in alt.split(",")]
            n = len(criterias)
            m = len(alternatives)
            A = np.ones((n, n))
            B = np.ones((n, m, m))

            # User input for pairwise comparison
            st.subheader("Perbandingan Berpasangan untuk Kriteria")
            for i in range(n):
                for j in range(i + 1, n):
                    A[i][j] = st.slider(
                        f"Kriteria {criterias[i]} dibandingkan dengan Kriteria {criterias[j]}",
                        1, 9, 1, key=f"crit_{i}_{j}"
                    )
                    A[j][i] = 1 / A[i][j]

            st.subheader("Perbandingan Berpasangan untuk Alternatif")
            for k in range(n):
                st.write(f"Alternatif untuk Kriteria {criterias[k]}")
                for i in range(m):
                    for j in range(i + 1, m):
                        B[k][i][j] = st.slider(
                            f"{alternatives[i]} dibandingkan dengan {alternatives[j]} untuk {criterias[k]}",
                            1, 9, 1, key=f"alt_{k}_{i}_{j}"
                        )
                        B[k][j][i] = 1 / B[k][i][j]

            if st.button("Hitung dan Simpan AHP"):
                if not respondent_name:
                    st.error("Mohon isi nama responden!")
                else:
                    # Perform AHP calculation and save the response
                    W, W2, W3 = calculate_ahp(A, B, n, m, criterias, alternatives)
                    save_response(respondent_name, A, B, W, criterias, alternatives)
                    st.success("Data berhasil dihitung dan disimpan!")

                    # Show results
                    df_result = pd.DataFrame({
                        'Alternatif': alternatives,
                        'Skor Akhir': W
                    })
                    df_result = df_result.sort_values('Skor Akhir', ascending=False).reset_index(drop=True)
                    df_result['Ranking'] = df_result['Skor Akhir'].rank(ascending=False).astype(int)
                    st.write("### Hasil Akhir AHP dengan Ranking:")
                    st.table(df_result)

    with tab2:
        st.header("Analisis Semua Responden")
        if st.session_state.responses:
            st.write("### Daftar Responden:")
            resp_df = pd.DataFrame([
                {'Nama': r['respondent_name'], 'Waktu': r['timestamp']}
                for r in st.session_state.responses
            ])
            st.dataframe(resp_df)
        else:
            st.info("Belum ada data responden yang tersimpan")

if __name__ == '__main__':
    main()
