import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# Initialize session state for storing responses
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Save response to session state
def save_response(respondent_name, respondent_role, A, B, W, criterias, alternatives):
    """
    Save the respondent's input data, including criteria and alternative weights,
    into the session state and CSV file.
    """
    response_data = {
        'respondent_name': str(respondent_name),
        'respondent_role': str(respondent_role),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'criteria_data': A.tolist(),
        'alternative_data': B.tolist(),
        'final_scores': [float(score) for score in W],  # Ensure float type
        'criterias': list(criterias),
        'alternatives': list(alternatives)
    }
    
    st.session_state.responses.append(response_data)
    save_to_csv()

def save_to_csv():
    """
    Save all responses in session state to a CSV file. Uses JSON serialization
    to handle complex data structures like lists and numpy arrays.
    """
    if len(st.session_state.responses) > 0:
        # Convert response data to DataFrame-friendly format
        df_data = []
        for response in st.session_state.responses:
            row_data = {
                'respondent_name': response['respondent_name'],
                'respondent_role': response['respondent_role'],
                'timestamp': response['timestamp'],
                'criteria_data': json.dumps(response['criteria_data']),
                'alternative_data': json.dumps(response['alternative_data']),
                'final_scores': json.dumps(response['final_scores']),
                'criterias': json.dumps(response['criterias']),
                'alternatives': json.dumps(response['alternatives'])
            }
            df_data.append(row_data)
        
        df = pd.DataFrame(df_data)
        df.to_csv('responses.csv', index=False)

def load_from_csv():
    """
    Load responses from a CSV file into session state. Handles deserialization of
    JSON strings back into Python lists and arrays.
    """
    try:
        df = pd.read_csv('responses.csv')
        # Convert string representations of lists back to actual lists/arrays
        for index, row in df.iterrows():
            response_data = {
                'respondent_name': str(row['respondent_name']),
                'respondent_role': str(row['respondent_role']),
                'timestamp': str(row['timestamp']),
                'criteria_data': json.loads(row['criteria_data']),
                'alternative_data': json.loads(row['alternative_data']),
                'final_scores': json.loads(row['final_scores']),
                'criterias': json.loads(row['criterias']),
                'alternatives': json.loads(row['alternatives'])
            }
            st.session_state.responses.append(response_data)
    except FileNotFoundError:
        st.session_state.responses = []
    except Exception as e:
        st.error(f"Error loading responses: {str(e)}")
        st.session_state.responses = []

@st.cache_data
def get_weight(A, str_label, labels):
    """
    Calculate normalized eigenvector (weights) from the pairwise comparison matrix A.
    Also checks the Consistency Ratio (CR) and warns if it's too high.
    """
    n = A.shape[0]
    e_vals, e_vecs = np.linalg.eig(A)
    lamb = np.max(np.real(e_vals))
    w = np.real(e_vecs[:, np.argmax(np.real(e_vals))])
    w = w / np.sum(w)  # Normalize weights
    
    # Consistency Index (CI) and Ratio (CR) calculation
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
    """
    Create a bar chart to visualize weights.
    """
    fig, ax = plt.subplots()
    ax.bar(y, x, color='#088eff')
    ax.set_facecolor('#F0F2F6')
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Nilai")
    plt.xticks(rotation=45)
    return fig

@st.cache_data
def calculate_ahp(A, B, n, m, criterias, alternatives):
    """
    Calculate the AHP scores by normalizing matrices and performing matrix multiplication.
    """
    # Ensure symmetry in the criteria matrix
    for i in range(n):
        for j in range(i, n):
            if i != j:
                A[j][i] = float(1 / A[i][j])
    dfA = pd.DataFrame(A, index=criterias, columns=criterias)
    st.markdown(" #### Tabel Kriteria")
    st.table(dfA)

    # Ensure symmetry in the alternatives matrices
    for k in range(n):
        for i in range(m):
            for j in range(i, m):
                if i != j:
                    B[k][j][i] = float(1 / B[k][i][j])
    
    st.write("---")

    # Calculate weights for criteria and alternatives
    W2 = get_weight(A, "Tabel Kriteria", criterias)
    W3 = np.zeros((n, m))

    for i in range(n):
        w3 = get_weight(B[i], f"Tabel Alternatif untuk Kriteria {criterias[i]}", alternatives)
        W3[i] = w3

    W = np.dot(W2, W3)
    return W, W2, W3

def calculate_aggregate_results(responses):
    """
    Calculate aggregate results from all responses with proper type conversion.
    """
    try:
        # Convert scores to proper numpy arrays
        all_scores = []
        for response in responses:
            scores = np.array(response['final_scores'], dtype=float)
            all_scores.append(scores)
        
        # Calculate mean only if we have scores
        if all_scores:
            all_scores_array = np.array(all_scores)
            avg_scores = np.mean(all_scores_array, axis=0)
            return avg_scores.tolist()
        return None
    except Exception as e:
        st.error(f"Error calculating aggregate results: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Kalkulator AHP", page_icon=":bar_chart:")
    st.header("Kalkulator AHP Untuk Menentukan Jenis Gamifikasi Pop-Up Campaign")
    
    # Try to load existing responses
    load_from_csv()
    
    # Add tabs for input and analysis
    tab1, tab2 = st.tabs(["Input Data", "Analisis Responden"])
    
    with tab1:
        st.sidebar.title("Kriteria & Alternatif")
        
        # Add respondent information fields
        respondent_name = st.text_input("Nama Responden")
        respondent_role = st.selectbox("Peran/Jabatan", 
            ["Manager", "Supervisor", "Staff", "Analyst", "Other"])
        
        st.sidebar.info("""
        ### Petunjuk Pengisian AHP
        
        Untuk mendapatkan hasil yang optimal dan konsisten, harap perhatikan langkah-langkah berikut saat mengisi nilai perbandingan:
        
        1. Masukkan Input Metrik dan Nama Jenis Gamifikasi dengan tanda , misal CTR, CR, IMPRESSION.
        2. **Konsistensi**: Jika Kriteria A lebih penting dari Kriteria B, dan Kriteria B lebih penting dari Kriteria C, maka Kriteria A seharusnya jauh lebih penting daripada Kriteria C.
        
        3. **Skala Pengisian**: Gunakan skala **1 hingga 9**:
           - 1: Sama penting
           - 3: Sedikit lebih penting
           - 5: Lebih penting
           - 7: Sangat lebih penting
           - 9: Mutlak lebih penting
        
        4. **Perbandingan Simetris**: Jika Anda menilai Kriteria A lebih penting daripada Kriteria B, maka sebaliknya, nilai Kriteria B terhadap Kriteria A harus otomatis terbalik.
        """)
        
        cri = st.sidebar.text_input("Masukkan Kriteria Metrik")
        alt = st.sidebar.text_input("Masukkan Alternatif Jenis Gamifikasi")
        
        if cri and alt:
            criterias = [c.strip() for c in cri.split(",")]
            alternatives = [a.strip() for a in alt.split(",")]
            
            n = len(criterias)
            m = len(alternatives)
            
            with st.expander("Bobot Kriteria"):
                st.subheader("Perbandingan Berpasangan untuk Kriteria")
                A = np.ones((n, n))
                
                for i in range(n):
                    for j in range(i+1, n):
                        st.markdown(f"##### Kriteria {criterias[i]} dibandingkan dengan Kriteria {criterias[j]}")
                        criteriaradio = st.radio(
                            f"Pilih kriteria yang lebih prioritas",
                            (criterias[i], criterias[j]),
                            key=f"crit_{i}_{j}",
                            horizontal=True
                        )
                        
                        if criteriaradio == criterias[i]:
                            A[i][j] = st.slider(
                                f"Seberapa jauh {criterias[i]} lebih penting dibandingkan {criterias[j]}?",
                                1, 9, 1, key=f"crit_slider_{i}_{j}"
                            )
                            A[j][i] = 1/A[i][j]
                        else:
                            A[j][i] = st.slider(
                                f"Seberapa jauh {criterias[j]} lebih penting dibandingkan {criterias[i]}?",
                                1, 9, 1, key=f"crit_slider_{j}_{i}"
                            )
                            A[i][j] = 1/A[j][i]
            
            with st.expander("Bobot Alternatif"):
                st.subheader("Perbandingan Berpasangan untuk Alternatif")
                B = np.ones((n, m, m))
                
                for k in range(n):
                    st.write("---")
                    st.markdown(f"##### Perbandingan Alternatif untuk Kriteria {criterias[k]}")
                    
                    for i in range(m):
                        for j in range(i+1, m):
                            alternativeradio = st.radio(
                                f"Pilih alternatif yang lebih prioritas untuk kriteria {criterias[k]}",
                                (alternatives[i], alternatives[j]),
                                key=f"alt_{k}_{i}_{j}",
                                horizontal=True
                            )
                            
                            if alternativeradio == alternatives[i]:
                                B[k][i][j] = st.slider(
                                    f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[i]} lebih baik dibandingkan {alternatives[j]}?",
                                    1, 9, 1, key=f"alt_slider_{k}_{i}_{j}"
                                )
                                B[k][j][i] = 1/B[k][i][j]
                            else:
                                B[k][j][i] = st.slider(
                                    f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[j]} lebih baik dibandingkan {alternatives[i]}?",
                                    1, 9, 1, key=f"alt_slider_{k}_{j}_{i}"
                                )
                                B[k][i][j] = 1/B[k][j][i]
            
            if st.button("Hitung dan Simpan AHP"):
                if not respondent_name:
                    st.error("Mohon isi nama responden terlebih dahulu!")
                else:
                    W, W2, W3 = calculate_ahp(A, B, n, m, criterias, alternatives)
                    
                    # Save response
                    save_response(respondent_name, respondent_role, A, B, W, criterias, alternatives)
                    
                    # Show results
                    df_result = pd.DataFrame({
                        'Alternatif': alternatives,
                        'Skor Akhir': W
                    })
                    df_result = df_result.sort_values('Skor Akhir', ascending=False).reset_index(drop=True)
                    df_result['Ranking'] = df_result['Skor Akhir'].rank(ascending=False).astype(int)
                    
                    st.pyplot(plot_graph(W2, criterias, "Kriteria", "Bobot Kriteria"))
                    st.pyplot(plot_graph(W, alternatives, "Alternatif", "Alternatif Optimal"))
                    
                    st.write("### Hasil Akhir AHP dengan Ranking:")
                    st.table(df_result[['Alternatif', 'Skor Akhir', 'Ranking']])
                    st.balloons()
    
    with tab2:
        st.header("Analisis Semua Responden")
        
        if len(st.session_state.responses) > 0:
            # Display respondents
            st.write("### Daftar Responden:")
            resp_df = pd.DataFrame([
                {
                    'Nama': r['respondent_name'],
                    'Peran': r['respondent_role'],
                    'Waktu': r['timestamp']
                }
                for r in st.session_state.responses
            ])
            st.dataframe(resp_df)
            
            # Add a reset button to clear all responses
            if st.button("Reset Data"):
                st.session_state.responses = []  # Clear session state
                try:
                    os.remove('responses.csv')  # Remove the CSV file if it exists
                    st.success("Data berhasil dihapus dan direset!")
                except FileNotFoundError:
                    st.warning("File CSV tidak ditemukan, tetapi data sudah direset.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menghapus file: {str(e)}")
            
            # Calculate aggregate results using the new function
            avg_scores = calculate_aggregate_results(st.session_state.responses)
            
            if avg_scores is not None:
                alternatives = st.session_state.responses[0]['alternatives']
                
                df_aggregate = pd.DataFrame({
                    'Alternatif': alternatives,
                    'Rata-rata Skor': avg_scores
                })
                df_aggregate = df_aggregate.sort_values('Rata-rata Skor', ascending=False)
                df_aggregate['Ranking'] = df_aggregate['Rata-rata Skor'].rank(ascending=False).astype(int)
                
                st.write("### Hasil Agregasi:")
                st.table(df_aggregate)
                
                st.pyplot(plot_graph(
                    avg_scores,
                    alternatives,
                    "Alternatif",
                    "Rata-rata Skor Alternatif dari Semua Responden"
                ))
                
                # Add download button for results
                csv = df_aggregate.to_csv(index=False)
                st.download_button(
                    "Download Hasil Agregasi (CSV)",
                    csv,
                    "ahp_results.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.error("Terjadi kesalahan dalam menghitung hasil agregasi")
        else:
            st.info("Belum ada data responden yang tersimpan")

if __name__ == '__main__':
    main()
