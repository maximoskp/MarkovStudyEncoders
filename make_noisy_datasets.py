from data_utils import generate_P_h_given_h, \
    generate_P_m_given_h, generate_dataset_pickle, HM_Dataset

h_size = 32
m_size = 64
seq_len = 128

num_samples = {
    'train': 10000,
    'test': 500
}
trans_peaks = {
    'T2': 2,
    'T10': 10
}
m_given_h_peaks = {
    'M2': 2,
    'M10': 10
}

for samples_name, samples_num in num_samples.items():
    for trans_name, trans_num in trans_peaks.items():
        for m_name, m_num in m_given_h_peaks.items():
            file_name = '_'.join([samples_name, 'n', trans_name, m_name])
            print(file_name)
            # Generate transition matrices
            h_token_ids, T_h = generate_P_h_given_h(
                num_peaks=trans_num,
                peaks_probability=0.3,
                h_size=h_size,
                seed=42
            )
            P_m_h, m_token_ids = generate_P_m_given_h(
                num_peaks=m_num,
                peaks_probability=0.3,
                m_size=m_size,
                h_token_ids=h_token_ids,
                seed=42
            )
            # Generate dataset pickle
            generate_dataset_pickle('data/' + file_name + '.pkl', num_samples=samples_num, 
                                    h_token_ids=h_token_ids, m_token_ids=m_token_ids, 
                                    T_h=T_h, P_m_h=P_m_h, seq_len=seq_len)