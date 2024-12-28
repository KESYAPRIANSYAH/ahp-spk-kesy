import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO, StringIO 
import json

# Initialize session state for storing responses
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Save response to session state and CSV
def save_response(name, A, B, criterias, alternatives, final_scores):
    response_data = {
        'name': name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'criteria_matrix': A.tolist(),
        'alternatives_matrix': B.tolist(),
        'criteria_list': criterias,
        'alternatives_list': alternatives,
        'final_scores': final_scores.tolist()
    }
    
    # Add to session state
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    st.session_state.responses.append(response_data)
    
    # Convert to DataFrame and save to CSV
    try:
        # Read existing CSV if it exists
        df_existing = pd.read_csv('responses.csv')
        df_new = pd.DataFrame([response_data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv('responses.csv', index=False)
    except FileNotFoundError:
        # Create new CSV if it doesn't exist
        df = pd.DataFrame([response_data])
        df.to_csv('responses.csv', index=False)

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
    
    st.write(f"### Normalized Eigenvector for {str_label}:")
    df_weight = pd.DataFrame(w, columns=['Weight'], index=labels)
    st.table(df_weight)
    
    st.write('CR = %f' % cr)
    if cr > 0.1:
        st.error(f"⚠️ Consistency check failed for {str_label}")

    return w

def plot_graph(x, y, ylabel, title):
    fig, ax = plt.subplots()
    ax.bar(y, x, color='#088eff')
    ax.set_facecolor('#F0F2F6')
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Value")
    plt.xticks(rotation=45)
    return fig

def delete_response(index):
    if 'responses' in st.session_state:
        del st.session_state.responses[index]
        # Update CSV file
        if st.session_state.responses:
            pd.DataFrame(st.session_state.responses).to_csv('responses.csv', index=False)
        else:
            # If no responses left, create empty CSV
            pd.DataFrame(columns=['name', 'timestamp', 'criteria_matrix', 'alternatives_matrix', 
                                'criteria_list', 'alternatives_list', 'final_scores']).to_csv('responses.csv', index=False)
            
@st.cache_data
def calculate_ahp(A, B, n, m, criterias, alternatives):
    for i in range(n):
        for j in range(i, n):
            if i != j:
                A[j][i] = float(1 / A[i][j])
    dfA = pd.DataFrame(A, index=criterias, columns=criterias)
    st.markdown(" #### Criteria")
    st.table(dfA)

    for k in range(n):
        for i in range(m):
            for j in range(i, m):
                if i != j:
                    B[k][j][i] = float(1 / B[k][i][j])
    st.write("---")

    for i in range(n):
        dfB = pd.DataFrame(B[i], index=alternatives, columns=alternatives)
        st.markdown(f" #### Alternatives for Criterion {criterias[i]}")
        st.table(dfB)

    W2 = get_weight(A, "Criteria", criterias)
    W3 = np.zeros((n, m))

    for i in range(n):
        w3 = get_weight(B[i], f"Alternatives for Criterion {criterias[i]}", alternatives)
        W3[i] = w3

    W = np.dot(W2, W3)
    
    df_result = pd.DataFrame({'Alternative': alternatives, 'Final Score': W})
    df_result = df_result.sort_values('Final Score', ascending=False).reset_index(drop=True)
    df_result['Ranking'] = df_result['Final Score'].rank(ascending=False).astype(int)

    # Plot AHP results
    st.pyplot(plot_graph(W2, criterias, "Criteria", "Criteria Weights"))
    st.pyplot(plot_graph(W, alternatives, "Alternatives", "Optimal Alternative for Given Criteria"))
    st.balloons()

    # Display Final Results with Ranking
    st.write("### Final AHP Results with Ranking:")
    st.table(df_result[['Alternative', 'Final Score', 'Ranking']])
    
    return W

def main():
    st.set_page_config(page_title="AHP Multi-Respondent Calculator", page_icon=":bar_chart:")
    st.header("AHP Calculator to Determine Gamification Types for Pop-Up Campaign")
    
    # Add tabs for input and analysis
    tab1, tab2 = st.tabs(["Input Data", "Respondent Analysis"])
    
    with tab1:
        # Respondent Information
        st.subheader("Respondent Information")
        name = st.text_input("Full Name")
        
        st.sidebar.title("Criteria & Alternatives")
        
        # Instructions in sidebar
        st.sidebar.info("""
        ### AHP Filling Instructions
        
        To obtain optimal and consistent results, please follow these steps when filling in comparison values:
        
        1. Enter metrics and gamification types separated by a comma, e.g., CTR, CR, IMPRESSION.
        2. **Consistency**: If Criterion A is more important than Criterion B, and Criterion B is more important than Criterion C, then Criterion A should be much more important than Criterion C.
        
        3. **Filling Scale**: Use a scale of **1 to 9**:
           - 1: Equally important
           - 3: Slightly more important
           - 5: More important
           - 7: Much more important
           - 9: Absolutely more important
        
        4. **Symmetric Comparison**: If you rate Criterion A as more important than Criterion B, the inverse should automatically apply for Criterion B compared to Criterion A.

        ### Usage of Values 2, 4, 6, and 8:
        - **Value 2**: Criterion A is slightly more important than Criterion B.
        - **Value 4**: Criterion A is more important than Criterion B, but not significantly.
        - **Value 6**: Criterion A is considerably more important than Criterion B.
        - **Value 8**: Criterion A is very much more important than Criterion B.
        """)
        
        cri = st.sidebar.text_input("Enter Criteria Metrics")
        alt = st.sidebar.text_input("Enter Gamification Alternatives")
        criterias = cri.split(",") if cri else []
        alternatives = alt.split(",") if alt else []

        if cri and alt and name:
            with st.expander("Criteria Weights"):
                st.subheader("Pairwise Comparison for Criteria")
                n = len(criterias)
                A = np.zeros((n, n))

                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            A[i][j] = 1
                        else:
                            st.markdown(f" ##### Criterion {criterias[i]} compared to Criterion {criterias[j]}")
                            criteriaradio = st.radio(
                                "Select the more prioritized criterion",
                                (criterias[i], criterias[j]),
                                key=f"crit_{i}_{j}",
                                horizontal=True
                            )

                            if criteriaradio == criterias[i]:
                                A[i][j] = st.slider(
                                    f"How much more important is {criterias[i]} compared to {criterias[j]}?",
                                    1, 9, 1, key=f"crit_slider_{i}_{j}"
                                )
                                A[j][i] = float(1/A[i][j])
                            else:
                                A[j][i] = st.slider(
                                    f"How much more important is {criterias[j]} compared to {criterias[i]}?",
                                    1, 9, 1, key=f"crit_slider_{j}_{i}"
                                )
                                A[i][j] = float(1/A[j][i])

            with st.expander("Alternative Weights"):
                st.subheader("Pairwise Comparison for Alternatives")
                m = len(alternatives)
                B = np.zeros((n, m, m))

                for k in range(n):
                    st.write("---")
                    st.markdown(f" ##### Comparison of Alternatives for Criterion {criterias[k]}")

                    for i in range(m):
                        for j in range(i, m):
                            if i == j:
                                B[k][i][j] = 1
                            else:
                                alternativeradio = st.radio(
                                    f"Select the more prioritized alternative for Criterion {criterias[k]}",
                                    (alternatives[i], alternatives[j]),
                                    key=f"alt_{k}_{i}_{j}",
                                    horizontal=True
                                )

                                if alternativeradio == alternatives[i]:
                                    B[k][i][j] = st.slider(
                                        f"Considering Criterion {criterias[k]}, how much better is {alternatives[i]} compared to {alternatives[j]}?",
                                        1, 9, 1, key=f"alt_slider_{k}_{i}_{j}"
                                    )
                                    B[k][j][i] = float(1/B[k][i][j])
                                else:
                                    B[k][j][i] = st.slider(
                                        f"Considering Criterion {criterias[k]}, how much better is {alternatives[j]} compared to {alternatives[i]}?",
                                        1, 9, 1, key=f"alt_slider_{k}_{j}_{i}"
                                    )
                                    B[k][i][j] = float(1/B[k][j][i])

            btn = st.button("Calculate and Save AHP")
            st.write("##")

            if btn:
                W = calculate_ahp(A, B, n, m, criterias, alternatives)
                save_response(name, A, B, criterias, alternatives, W)
                st.success("Data successfully saved!")
    
    with tab2:
        st.subheader("Multi-Respondent Analysis")
        
        if st.session_state.responses:
            # Display list of respondents
            st.write("### Respondent List:")
            for idx, resp in enumerate(st.session_state.responses):
                with st.expander(f"Respondent: {resp['name']} - {resp['timestamp']}"):
                    col1, col2 = st.columns([3,1])
                    with col1:
                        st.write("Final Scores:")
                        df_result = pd.DataFrame({
                            'Alternative': resp['alternatives_list'],
                            'Score': resp['final_scores']
                        })
                        st.table(df_result)
                        
                        # Download button for individual response
                        individual_resp = pd.DataFrame([resp])
                        individual_resp['criteria_matrix'] = individual_resp['criteria_matrix'].apply(lambda x: np.array(x).tolist())
                        individual_resp['alternatives_matrix'] = individual_resp['alternatives_matrix'].apply(lambda x: np.array(x).tolist())
                        individual_resp['final_scores'] = individual_resp['final_scores'].apply(lambda x: np.array(x).tolist())
                        csv_individual = individual_resp.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download Data {resp['name']}",
                            data=csv_individual,
                            file_name=f"ahp_response_{resp['name'].lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        if st.button("Delete Data", key=f"delete_{idx}"):
                            delete_response(idx)
                            st.experimental_rerun()
            
            # Calculate and display average scores
            avg_scores = calculate_average_scores(st.session_state.responses, alternatives)
            if avg_scores is not None and alternatives:
                st.write("### Average Scores Across All Respondents:")
                df_avg = pd.DataFrame({
                    'Alternative': alternatives,
                    'Average Score': avg_scores
                })
                # Sort and add ranking
                df_avg = df_avg.sort_values('Average Score', ascending=False).reset_index(drop=True)
                df_avg['Ranking'] = df_avg['Average Score'].rank(ascending=False).astype(int)
                
                # Display table with ranking
                st.table(df_avg[['Alternative', 'Average Score', 'Ranking']])
                
                # Download button for average scores
                csv_avg = df_avg.to_csv(index=False)
                st.download_button(
                    label="📥 Download Average Scores",
                    data=csv_avg,
                    file_name="ahp_average_scores.csv",
                    mime="text/csv",
                    help="Download average scores in CSV format"
                )
                
                # Plot average scores
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(df_avg['Alternative'], df_avg['Average Score'], color='#088eff')
                
                # Add ranking labels on top of bars
                for idx, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'Rank {df_avg.iloc[idx]["Ranking"]}',
                           ha='center', va='bottom')
                
                ax.set_title("Average Scores of Alternatives Across All Respondents")
                ax.set_xlabel("Alternative")
                ax.set_ylabel("Average Score")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download button for plot
                st.write("### Download Plot")
                # Use a new buffer for each plot
                plot_buffer = BytesIO()
                plt.savefig(plot_buffer, format="png", dpi=300, bbox_inches='tight')
                st.download_button(
                    label="📥 Download Plot (PNG)",
                    data=plot_buffer.getvalue(),
                    file_name="ahp_average_scores_plot.png",
                    mime="image/png",
                    help="Download plot in PNG format"
                )
        else:
            st.info("No respondent data saved yet.")

if __name__ == '__main__':
    main()
