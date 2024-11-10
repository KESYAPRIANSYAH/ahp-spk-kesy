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
    save_to_csv()

def save_to_csv():
    if len(st.session_state.responses) > 0:
        df_data = [
            {
                'respondent_name': response['respondent_name'],
                'timestamp': response['timestamp'],
                'criteria_data': json.dumps(response['criteria_data']),
                'alternative_data': json.dumps(response['alternative_data']),
                'final_scores': json.dumps(response['final_scores']),
                'criterias': json.dumps(response['criterias']),
                'alternatives': json.dumps(response['alternatives'])
            }
            for response in st.session_state.responses
        ]
        df = pd.DataFrame(df_data)
        df.to_csv('responses.csv', index=False)

def load_from_csv():
    try:
        df = pd.read_csv('responses.csv')
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
    n = A.shape[0]
    e_vals, e_vecs = np.linalg.eig(A)
    lamb = np.max(np.real(e_vals))
    w = np.real(e_vecs[:, np.argmax(np.real(e_vals))])
    w = w / np.sum(w)

    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
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
    
    load_from_csv()

    tab1, tab2 = st.tabs(["Input Data", "Analisis Responden"])

    with tab1:
        st.sidebar.title("Kriteria & Alternatif")
        respondent_name = st.text_input("Nama Responden")
        
        cri = st.sidebar.text_input("Masukkan Kriteria Metrik")
        alt = st.sidebar.text_input("Masukkan Alternatif Jenis Gamifikasi")
        
        if cri and alt:
            criterias = [c.strip() for c in cri.split(",")]
            alternatives = [a.strip() for a in alt.split(",")]
            n, m = len(criterias), len(alternatives)

            A = np.ones((n, n))
            B = np.ones((n, m, m))
            
            with st.expander("Bobot Kriteria"):
                for i in range(n):
                    for j in range(i+1, n):
                        criteriaradio = st.radio(
                            f"Pilih kriteria yang lebih prioritas",
                            (criterias[i], criterias[j]),
                            key=f"crit_{i}_{j}",
                            horizontal=True
                        )
                        if criteriaradio == criterias[i]:
                            A[i][j] = st.slider(f"Seberapa jauh {criterias[i]} lebih penting dibandingkan {criterias[j]}?", 1, 9, 1, key=f"crit_slider_{i}_{j}")
                            A[j][i] = 1 / A[i][j]
                        else:
                            A[j][i] = st.slider(f"Seberapa jauh {criterias[j]} lebih penting dibandingkan {criterias[i]}?", 1, 9, 1, key=f"crit_slider_{j}_{i}")
                            A[i][j] = 1 / A[j][i]

            with st.expander("Bobot Alternatif"):
                for k in range(n):
                    for i in range(m):
                        for j in range(i+1, m):
                            alternativeradio = st.radio(
                                f"Pilih alternatif yang lebih prioritas untuk kriteria {criterias[k]}",
                                (alternatives[i], alternatives[j]),
                                key=f"alt_{k}_{i}_{j}",
                                horizontal=True
                            )
                            if alternativeradio == alternatives[i]:
                                B[k][i][j] = st.slider(f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[i]} lebih baik dibandingkan {alternatives[j]}?", 1, 9, 1, key=f"alt_slider_{k}_{i}_{j}")
                                B[k][j][i] = 1 / B[k][i][j]
                            else:
                                B[k][j][i] = st.slider(f"Dengan mempertimbangkan Kriteria {criterias[k]}, seberapa jauh {alternatives[j]} lebih baik dibandingkan {alternatives[i]}?", 1, 9, 1, key=f"alt_slider_{k}_{j}_{i}")
                                B[k][i][j] = 1 / B[k][j][i]

            if st.button("Simpan Data"):
                if respondent_name:
                    save_response(respondent_name, A, B, [], criterias, alternatives)
                    st.success("Data berhasil disimpan!")
                else:
                    st.error("Mohon isi nama responden terlebih dahulu!")
            
            if st.button("Hitung AHP"):
                if respondent_name:
                    W, W2, W3 = calculate_ahp(A, B, n, m, criterias, alternatives)
                    st.pyplot(plot_graph(W2, criterias, "Kriteria", "Bobot Kriteria"))
                    st.pyplot(plot_graph(W, alternatives, "Alternatif", "Alternatif Optimal"))
                    st.write("### Hasil Akhir AHP dengan Ranking:")
                    df_result = pd.DataFrame({'Alternatif': alternatives, 'Skor Akhir': W})
                    df_result = df_result.sort_values('Skor Akhir', ascending=False).reset_index(drop=True)
                    df_result['Ranking'] = df_result['Skor Akhir'].rank(ascending=False).astype(int)
                    st.table(df_result[['Alternatif', 'Skor Akhir', 'Ranking']])
                    st.balloons()
                else:
                    st.error("Mohon isi nama responden terlebih dahulu!")

    with tab2:
        if len(st.session_state.responses) > 0:
            st.write("### Daftar Responden:")
            resp_df = pd.DataFrame([
                {'Nama': r['respondent_name'], 'Waktu': r['timestamp']} 
                for r in st.session_state.responses
            ])
            st.dataframe(resp_df)

            selected_name = st.selectbox("Pilih Nama Responden untuk Dihapus", [r['respondent_name'] for r in st.session_state.responses])
            if st.button("Hapus Data Responden"):
                st.session_state.responses = [r for r in st.session_state.responses if r['respondent_name'] != selected_name]
                save_to_csv()
                st.success("Data berhasil dihapus!")

            if st.button("Reset Data"):
                st.session_state.responses = []
                try:
                    os.remove('responses.csv')
                    st.success("Data berhasil dihapus dan direset!")
                except FileNotFoundError:
                    st.warning("File CSV tidak ditemukan, tetapi data sudah direset.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menghapus file: {str(e)}")
        else:
            st.info("Belum ada data responden yang tersimpan")

if __name__ == '__main__':
    main()
