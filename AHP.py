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

@st.cache_data
def calculate_ahp(A, B, n, m, criterias, alternatives):
    # ... [previous calculate_ahp code remains the same] ...
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
        ... [previous instructions remain the same] ...
        """)
        
        cri = st.sidebar.text_input("Masukkan Kriteria Metrik")
        alt = st.sidebar.text_input("Masukkan Alternatif Jenis Gamifikasi")
        criterias = cri.split(",") if cri else []
        alternatives = alt.split(",") if alt else []

        if cri and alt and name:
            # ... [previous input collection code remains the same] ...

            btn = st.button("Hitung dan Simpan AHP")
            st.write("##")

            if btn:
                W = calculate_ahp(A, B, n, m, criterias, alternatives)
                save_response(name, A, B, criterias, alternatives, W)
                st.success(f"Data untuk responden {name} berhasil disimpan!")
    
    with tab2:
        st.subheader("Analisis Multi-Responden")
        
        # Add download button for all responses
        if st.session_state.responses:
            # Convert responses to DataFrame for download
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
        
        # Display all responses with delete buttons
        if st.session_state.responses:
            st.write("### Daftar Responden:")
            
            # Create columns for better layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                for resp in st.session_state.responses:
                    with st.expander(f"Responden: {resp['name']} - {resp['timestamp']}"):
                        st.write("Skor Akhir:")
                        df_result = pd.DataFrame({
                            'Alternatif': resp['alternatives_list'],
                            'Skor': resp['final_scores']
                        })
                        st.table(df_result)
                        
                        # Add delete button for each response
                        if st.button(f"Hapus Data {resp['name']}", key=f"delete_{resp['id']}"):
                            delete_response(resp['id'])
                            st.success(f"Data {resp['name']} berhasil dihapus!")
                            st.experimental_rerun()
            
            with col2:
                # Calculate and display average scores
                avg_scores = calculate_average_scores(st.session_state.responses, alternatives)
                if avg_scores is not None and alternatives:
                    st.write("### Rata-rata Skor:")
                    df_avg = pd.DataFrame({
                        'Alternatif': alternatives,
                        'Rata-rata Skor': avg_scores
                    })
                    df_avg = df_avg.sort_values('Rata-rata Skor', ascending=False)
                    st.table(df_avg)
                    
                    # Plot average scores
                    fig, ax = plt.subplots()
                    ax.bar(df_avg['Alternatif'], df_avg['Rata-rata Skor'], color='#088eff')
                    ax.set_title("Rata-rata Skor Alternatif")
                    ax.set_xlabel("Alternatif")
                    ax.set_ylabel("Rata-rata Skor")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        else:
            st.info("Belum ada data responden yang tersimpan.")

if __name__ == '__main__':
    main()
