import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Initialize session state for storing responses
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Save response to session state
def save_response(name, A, B, criterias, alternatives, final_scores):
    response_data = {
        'name': name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'criteria_matrix': A.tolist(),
        'alternatives_matrix': B.tolist(),
        'criteria_list': criterias,
        'alternatives_list': alternatives,
        'final_scores': final_scores.tolist(),
        'id': len(st.session_state.responses)  # Add unique ID for deletion
    }
    
    # Add to session state
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    st.session_state.responses.append(response_data)

# Delete response by ID
def delete_response(response_id):
    st.session_state.responses = [r for r in st.session_state.responses if r['id'] != response_id]

# Calculate average scores across all respondents
def calculate_average_scores(responses, alternatives):
    if not responses:
        return None
    
    all_scores = [resp['final_scores'] for resp in responses]
    avg_scores = np.mean(all_scores, axis=0)
    return avg_scores

@st.cache_data
def get_weight(A, str_label, labels):
    n = A.shape[0]
    e_vals, e_vecs = np.linalg.eig(A)
    lamb = np.max(np.real(e_vals))
    w = np.real(e_vecs[:, np.argmax(np.real(e_vals))])
    w = w / np.sum(w)
    
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24,
          7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
    ci = (lamb - n) / (n - 1)
    cr = ci / ri.get(n, float('inf'))
    
    st.write(f"### Vektor Eigen yang Dinormalisasi untuk {str_label}:")
    df_weight = pd.DataFrame(w, columns=['Bobot'], index=labels)
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
    plt.xticks(rotation=45)
    return fig

def display_detailed_matrices(resp):
    """Display detailed matrices for a single respondent"""
    # Display Criteria Matrix
    st.write("#### Tabel Kriteria")
    criteria_matrix = np.array(resp['criteria_matrix'])
    df_criteria = pd.DataFrame(
        criteria_matrix,
        index=resp['criteria_list'],
        columns=resp['criteria_list']
    )
    st.table(df_criteria)

    # Display Alternative Matrices for each criterion
    alternatives_matrix = np.array(resp['alternatives_matrix'])
    for i, criterion in enumerate(resp['criteria_list']):
        st.write(f"#### Tabel Alternatif untuk Kriteria {criterion}")
        df_alt = pd.DataFrame(
            alternatives_matrix[i],
            index=resp['alternatives_list'],
            columns=resp['alternatives_list']
        )
        st.table(df_alt)

def display_results_and_charts(resp):
    """Display final scores and charts for a single respondent"""
    st.write("#### Hasil Akhir")
    df_result = pd.DataFrame({
        'Alternatif': resp['alternatives_list'],
        'Skor': resp['final_scores']
    })
    df_result['Ranking'] = df_result['Skor'].rank(ascending=False).astype(int)
    df_result = df_result.sort_values('Skor', ascending=False)
    st.table(df_result)

    # Plot bar chart for final scores
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_result['Alternatif'], df_result['Skor'], color='#088eff')
    ax.set_title("Skor Akhir Alternatif")
    ax.set_xlabel("Alternatif")
    ax.set_ylabel("Skor")
    plt.xticks(rotation=45)
    st.pyplot(fig)
@st.cache_data
def calculate_ahp(A, B, n, m, criterias, alternatives):
    # Calculate criteria matrix
    for i in range(n):
        for j in range(i, n):
            if i != j:
                A[j][i] = float(1 / A[i][j])
    dfA = pd.DataFrame(A, index=criterias, columns=criterias)
    st.markdown(" #### Tabel Kriteria")
    st.table(dfA)

    # Calculate alternatives matrix
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

    # Calculate weights
    W2 = get_weight(A, "Tabel Kriteria", criterias)
    W3 = np.zeros((n, m))

    for i in range(n):
        w3 = get_weight(B[i], f"Tabel Alternatif untuk Kriteria {criterias[i]}", alternatives)
        W3[i] = w3

    # Calculate final scores
    W = np.dot(W2, W3)
    
    # Create results DataFrame
    df_result = pd.DataFrame({'Alternatif': alternatives, 'Skor Akhir': W})
    df_result = df_result.sort_values('Skor Akhir', ascending=False).reset_index(drop=True)
    df_result['Ranking'] = df_result['Skor Akhir'].rank(ascending=False).astype(int)

    # Plot graphs
    st.pyplot(plot_graph(W2, criterias, "Kriteria", "Bobot Kriteria"))
    st.pyplot(plot_graph(W, alternatives, "Alternatif", "Alternatif Optimal untuk Kriteria yang Diberikan"))
    st.balloons()

    # Display final results
    st.write("### Hasil Akhir AHP dengan Ranking:")
    st.table(df_result[['Alternatif', 'Skor Akhir', 'Ranking']])
    
    return W

def main():
    st.set_page_config(page_title="Kalkulator AHP Multi-Responden", page_icon=":bar_chart:", layout="wide")
    st.header("Kalkulator AHP Untuk Menentukan Jenis Gamifikasi Pop-Up Campaign")
    
    # Add tabs for input and analysis
    tab1, tab2 = st.tabs(["Input Data", "Analisis Responden"])
    
    with tab1:
        # Respondent Information
        st.subheader("Informasi Responden")
        name = st.text_input("Nama Lengkap")
        
        st.sidebar.title("Kriteria & Alternatif")
        
        # Instructions in sidebar
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
        criterias = cri.split(",") if cri else []
        alternatives = alt.split(",") if alt else []

        if cri and alt and name:
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
                            criteriaradio = st.radio(
                                "Pilih kriteria yang lebih prioritas ",
                                (criterias[i], criterias[j]),
                                key=f"crit_{i}_{j}",
                                horizontal=True
                            )

                            if criteriaradio == criterias[i]:
                                A[i][j] = st.slider(
                                    f"Seberapa jauh {criterias[i]} lebih penting dibandingkan {criterias[j]} ?",
                                    1, 9, 1, key=f"crit_slider_{i}_{j}"
                                )
                                A[j][i] = float(1/A[i][j])
                            else:
                                A[j][i] = st.slider(
                                    f"Seberapa jauh {criterias[j]} lebih penting dibandingkan {criterias[i]} ?",
                                    1, 9, 1, key=f"crit_slider_{j}_{i}"
                                )
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
                                alternativeradio = st.radio(
                                    f"Pilih alternatif yang lebih prioritas untuk kriteria {criterias[k]}",
                                    (alternatives[i], alternatives[j]),
                                    key=f"alt_{k}_{i}_{j}",
                                    horizontal=True
                                )

                                if alternativeradio == alternatives[i]:
                                    B[k][i][j] = st.slider(
                                        f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[i]} lebih baik dibandingkan {alternatives[j]} ?",
                                        1, 9, 1, key=f"alt_slider_{k}_{i}_{j}"
                                    )
                                    B[k][j][i] = float(1/B[k][i][j])
                                else:
                                    B[k][j][i] = st.slider(
                                        f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[j]} lebih baik dibandingkan {alternatives[i]} ?",
                                        1, 9, 1, key=f"alt_slider_{k}_{j}_{i}"
                                    )
                                    B[k][i][j] = float(1/B[k][j][i])

            btn = st.button("Hitung dan Simpan AHP")
            st.write("##")

            if btn:
                W = calculate_ahp(A, B, n, m, criterias, alternatives)
                save_response(name, A, B, criterias, alternatives, W)
                st.success(f"Data untuk responden {name} berhasil disimpan!")
with tab2:
        st.subheader("Analisis Multi-Responden")
        
        if st.session_state.responses:
            # Add download button for all responses
            df_download = pd.DataFrame([{
                'Nama': r['name'],
                'Waktu': r['timestamp'],
                'Kriteria': ', '.join(r['criteria_list']),
                'Alternatif': ', '.join(r['alternatives_list']),
                'Skor': ', '.join([str(s) for s in r['final_scores']])
            } for r in st.session_state.responses])
            
            csv = df_download.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Semua Data (CSV)",
                csv,
                "ahp_responses.csv",
                "text/csv",
                key='download-csv'
            )

            # Display aggregate analysis
            st.write("### Analisis Agregat")
            if len(st.session_state.responses) > 0:
                # Get the latest response for criteria and alternatives list
                latest_resp = st.session_state.responses[-1]
                alternatives = latest_resp['alternatives_list']
                
                # Calculate average scores
                avg_scores = calculate_average_scores(st.session_state.responses, alternatives)
                if avg_scores is not None:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("#### Rata-rata Skor Semua Responden")
                        df_avg = pd.DataFrame({
                            'Alternatif': alternatives,
                            'Rata-rata Skor': avg_scores
                        })
                        df_avg = df_avg.sort_values('Rata-rata Skor', ascending=False)
                        df_avg['Ranking'] = df_avg['Rata-rata Skor'].rank(ascending=False).astype(int)
                        st.table(df_avg)
                    
                    with col2:
                        # Plot average scores
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.bar(df_avg['Alternatif'], df_avg['Rata-rata Skor'], color='#088eff')
                        ax.set_title("Rata-rata Skor Alternatif dari Semua Responden")
                        ax.set_xlabel("Alternatif")
                        ax.set_ylabel("Rata-rata Skor")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

            # Display individual responses
            st.write("### Analisis Per Responden")
            for resp in st.session_state.responses:
                with st.expander(f"Responden: {resp['name']} - {resp['timestamp']}"):
                    tab_results, tab_matrices = st.tabs(["Hasil Akhir", "Matriks Detail"])
                    
                    with tab_results:
                        display_results_and_charts(resp)
                    
                    with tab_matrices:
                        display_detailed_matrices(resp)
                    
                    # Add delete button
                    if st.button(f"Hapus Data {resp['name']}", key=f"delete_{resp['id']}"):
                        delete_response(resp['id'])
                        st.success(f"Data {resp['name']} berhasil dihapus!")
                        st.experimental_rerun()
        else:
            st.info("Belum ada data responden yang tersimpan.")

if __name__ == '__main__':
    main()
