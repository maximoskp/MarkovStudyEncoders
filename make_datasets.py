from data_utils import generate_P_h_given_h, \
    generate_P_m_given_h, generate_dataset_pickle, HM_Dataset

h_size = 32
m_size = 64
seq_len = 128

num_samples = {
    'train': 10000,
    'test': 500
}
diagonal_probs = {
    'Thigh': 0.9,
    'Tlow': 0.2
}
num_peaks = {
    'P2': 2,
    'P10': 10
}
cross = {
    'Chigh': 0.9,
    'Clow': 0.2
}

for samples_name, samples_num in num_samples.items():
    for diag_name, diag_prob in diagonal_probs.items():
        for peaks_name, peaks_num in num_peaks.items():
            for cross_name, cross_probs in cross.items():
                file_name = '_'.join([samples_name, diag_name, peaks_name, cross_name])
                print(file_name)
                # Generate transition matrices
                h_token_ids, T_h = generate_P_h_given_h(diagonal_probability=diag_prob, h_size=h_size, seed=42)
                P_m_h, m_token_ids = generate_P_m_given_h(num_peaks=peaks_num, peaks_probability=cross_probs,
                                                        m_size=m_size, h_token_ids=h_token_ids, seed=42)
                # Generate dataset pickle
                generate_dataset_pickle('data/' + file_name + '.pkl', num_samples=samples_num, 
                                        h_token_ids=h_token_ids, m_token_ids=m_token_ids, 
                                        T_h=T_h, P_m_h=P_m_h, seq_len=seq_len)