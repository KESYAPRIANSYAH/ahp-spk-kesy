import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# Initialize session state for storing responses and current calculation
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'current_calculation' not in st.session_state:
    st.session_state.current_calculation = None

def save_response(respondent_name, calculation_data):
    """
    Save the respondent's calculation data into the session state and CSV file.
    """
    response_data = {
        'respondent_name': str(respondent_name),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'criteria_data': calculation_data['criteria_matrix'].tolist(),
        'alternative_data': calculation_data['alternative_matrix'].tolist(),
        'final_scores': [float(score) for score in calculation_data['final_scores']],
        'criterias': list(calculation_data['criterias']),
        'alternatives': list(calculation_data['alternatives'])
    }
    
    st.session_state.responses.append(response_data)
    save_to_csv()
    st.success("Data berhasil disimpan!")

def delete_response(index):
    """
    Delete a specific response from the session state and update the CSV file.
    """
    if 0 <= index < len(st.session_state.responses):
        del st.session_state.responses[index]
        save_to_csv()
        st.success("Data berhasil dihapus!")
        st.experimental_rerun()

def save_to_csv():
    """
    Save all responses in session state to a CSV file.
    """
    if len(st.session_state.responses) > 0:
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

def load_from_csv():
    """
    Load responses from a CSV file into session state.
    """
    try:
        df = pd.read_csv('responses.csv')
        st.session_state.responses = []
        for _, row in df.iterrows():
            response_data = {
                'respondent_name': str(row['respondent_name']),
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
    """
    n = A.shape[0]
    e_vals, e_vecs = np.linalg.eig(A)
    lamb = np.max(np.real(e_vals))
    w = np.real(e_vecs[:, np.argmax(np.real(e_vals))])
    w = w / np.sum(w)
    
    # Consistency calculation
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
    Calculate the AHP scores.
    """
    # Ensure symmetry in matrices
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

    W2 = get_weight(A, "Tabel Kriteria", criterias)
    W3 = np.zeros((n, m))

    for i in range(n):
        w3 = get_weight(B[i], f"Tabel Alternatif untuk Kriteria {criterias[i]}", alternatives)
        W3[i] = w3

    W = np.dot(W2, W3)
    
    return {
        'final_scores': W,
        'criteria_weights': W2,
        'alternative_weights': W3,
        'criteria_matrix': A,
        'alternative_matrix': B,
        'criterias': criterias,
        'alternatives': alternatives
    }

def display_results(calculation_data):
    """
    Display the AHP calculation results.
    """
    W = calculation_data['final_scores']
    W2 = calculation_data['criteria_weights']
    alternatives = calculation_data['alternatives']
    criterias = calculation_data['criterias']
    
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

def main():
    st.set_page_config(page_title="Kalkulator AHP", page_icon=":bar_chart:")
    st.header("Kalkulator AHP Untuk Menentukan Jenis Gamifikasi Pop-Up Campaign")
    
    load_from_csv()
    
    tab1, tab2 = st.tabs(["Input Data", "Analisis Responden"])
    
    with tab1:
        st.sidebar.title("Kriteria & Alternatif")
        
        respondent_name = st.text_input("Nama Responden")
        
        st.sidebar.info("""
        ### Petunjuk Pengisian AHP
        1. Masukkan kriteria dan alternatif yang akan dinilai.
        2. Gunakan skala 1 hingga 9 untuk menilai perbandingan.
        """)
        
        cri = st.sidebar.text_input("Masukkan Kriteria (pisahkan dengan koma)")
        alt = st.sidebar.text_input("Masukkan Alternatif (pisahkan dengan koma)")
        
        if cri and alt:
            criterias = [c.strip() for c in cri.split(",")]
            alternatives = [a.strip() for a in alt.split(",")]
            
            n = len(criterias)
            m = len(alternatives)
            
            # Input criteria comparison matrix
            A = np.ones((n, n))
            with st.expander("Bobot Kriteria"):
                st.subheader("Perbandingan Berpasangan untuk Kriteria")
                for i in range(n):
                    for j in range(i + 1, n):
                        st.markdown(f"##### Kriteria {criterias[i]} vs {criterias[j]}")
                        value = st.slider(f"{criterias[i]} dibandingkan dengan {criterias[j]}", 1, 9, 1)
                        A[i, j] = value
                        A[j, i] = 1 / value

            # Input alternatives comparison matrix
            B = np.ones((n, m, m))
            with st.expander("Bobot Alternatif"):
                st.subheader("Perbandingan Berpasangan untuk Alternatif")
                for k in range(n):
                    st.write(f"### Kriteria: {criterias[k]}")
                   
