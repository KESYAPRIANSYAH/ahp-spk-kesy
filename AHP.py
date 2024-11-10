import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# Initialize session state for storing responses and calculation results
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# Save response to session state
def save_response(respondent_name, A, B, W, criterias, alternatives):
    """
    Save the respondent's input data, including criteria and alternative weights,
    into the session state and CSV file, but only if the data is not duplicated.
    """
    response_data = {
        'respondent_name': str(respondent_name),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'criteria_data': A.tolist(),
        'alternative_data': B.tolist(),
        'final_scores': [float(score) for score in W],  # Ensure float type
        'criterias': list(criterias),
        'alternatives': list(alternatives)
    }
    
    # Check if the exact same response already exists
    if response_data not in st.session_state.responses:
        st.session_state.responses.append(response_data)
        save_to_csv()
        st.success("Data berhasil disimpan!")
    else:
        st.warning("Data sudah tersimpan sebelumnya dan tidak akan disimpan lagi.")

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

def main():
    st.set_page_config(page_title="Kalkulator AHP", page_icon=":bar_chart:")
    st.header("Kalkulator AHP Untuk Menentukan Jenis Gamifikasi Pop-Up Campaign")
    
    # Add input fields
    respondent_name = st.text_input("Nama Responden")
    
    st.sidebar.title("Kriteria & Alternatif")
    st.sidebar.info("""
    ### Petunjuk Pengisian AHP
    Harap isi nilai perbandingan dengan cermat dan gunakan skala 1 hingga 9.
    """)
    
    cri = st.sidebar.text_input("Masukkan Kriteria Metrik (pisahkan dengan koma)")
    alt = st.sidebar.text_input("Masukkan Alternatif Jenis Gamifikasi (pisahkan dengan koma)")
    
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
                        A[j][i] = 1 / A[i][j]
                    else:
                        A[j][i] = st.slider(
                            f"Seberapa jauh {criterias[j]} lebih penting dibandingkan {criterias[i]}?",
                            1, 9, 1, key=f"crit_slider_{j}_{i}"
                        )
                        A[i][j] = 1 / A[j][i]
        
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
                            B[k][j][i] = 1 / B[k][i][j]
                        else:
                            B[k][j][i] = st.slider(
                                f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[j]} lebih baik dibandingkan {alternatives[i]}?",
                                1, 9, 1, key=f"alt_slider_{k}_{j}_{i}"
                            )
                            B[k][i][j] = 1 / B[k][j][i]
        
        # Calculate AHP
        if st.button("Hitung AHP"):
            W, W2, W3 = calculate_ahp(A, B, n, m, criterias, alternatives)
            st.session_state.current_result = {
                'respondent_name': respondent_name,
                'criteria_data': A,
                'alternative_data': B,
                'final_scores': W,
                'criterias': criterias,
                'alternatives': alternatives
            }
            
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
        
        # Save result if it has been calculated
        if st.session_state.current_result and st.button("Simpan Hasil"):
            result = st.session_state.current_result
            save_response(
                result['respondent_name'],
                result['criteria_data'],
                result['alternative_data'],
                result['final_scores'],
                result['criterias'],
                result['alternatives']
            )

if __name__ == '__main__':
    main()
