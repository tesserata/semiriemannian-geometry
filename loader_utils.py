# loader utils
import pandas as pd
import os
import numpy as np
from scipy.sparse import csr_matrix
import scipy
from sklearn.preprocessing import LabelEncoder

# to_read = "/home/daria/neuro/connectomics/ADNI/data/"
# p_path = "/home/daria/neuro/connectomics/ADNI/data/matrices/"


to_read = "./data/"
p_path = "./data/matrices/"
xyz_path = "%s/adni2_centers/" % to_read
meta = pd.read_excel(to_read + 'ADNI2_Master_Subject_List.xls',
                     sheetname='Subject List', index_col=0)
names = meta['Subject ID'].unique()


def load_data_oftypes(dtypes, subject_names=names):
    # if dtypes[0] not in ['Normal', 'AD', 'EMCI', 'LMCI'] or dtypes[1] not in ['Normal', 'AD', 'EMCI', 'LMCI']:
    #     raise ValueError('dtypes must be Normal, AD, EMCI or LMCI')
    diagnosis = []
    names = []
    for name in subject_names:
        if any(str(name) in s for s in sorted(os.listdir(p_path))):
            ds = meta.loc[meta['Subject ID'] == name]["DX Group"].values
            d = np.unique(ds[np.array([isinstance(e, str) for e in ds])])
            if d.shape[0] == 1:
                for i in range(len(dtypes)):
                    if dtypes[i] in d[0]:
                        diagnosis.append(i)
                        names.append(name)

    full_orig_matrices_adni = []  # !!!
    xyz_adni = []
    target_long = []
    id_list = []

    extra_folders = ['003_S_4644_1', '003_S_5187_1', '127_S_5056_1',
                     '127_S_5058_1', '127_S_4992_1', '007_S_2106_1',
                     '007_S_2394_3', '021_S_4245_2', '098_S_2079_3',
                     '109_S_4531_4', '127_S_2234_2', '127_S_2234_6',
                     '127_S_4240_5', '127_S_4240_5']

    for i in range(len(diagnosis)):
        name = names[i]
        curr_diagnosis = diagnosis[i]
        for folder in sorted(os.listdir(p_path)):
            if str(name) in folder and folder not in extra_folders:
                id_list.append(name)
                full_path = p_path + folder
                target_long.append(curr_diagnosis)
                for filename in sorted(os.listdir(full_path)):
                    if 'NORM' not in filename:
                        m_df = pd.read_csv('%s/%s' % (full_path, filename), header=None, sep='\t')
                        m = np.delete(m_df.values, [3, 38], 0)
                        m = np.delete(m, [3, 38], 1)
                        m = np.array(m, dtype=float)
                        m = m - np.diag(np.diag(m))
                        full_orig_matrices_adni.append(m)
                full_xyz = '%s/%s_connect_dil_centerOfGravities.txt' % (xyz_path, folder)
                coord = pd.read_csv(full_xyz)
                coord = coord.values[:, 1:4]
                coord = np.delete(coord, [3, 38], 0)
                xyz_adni.append(coord)

    xyz_inv_matrices_adni = []  # !!!
    for s in range(len(full_orig_matrices_adni)):
        A = full_orig_matrices_adni[s]
        C = xyz_adni[s]
        dim = A.shape[0]
        D = np.zeros((dim, dim))
        for i in range(0, A.shape[0]):
            for j in range(i + 1, A.shape[0]):
                if A[i, j] != 0:
                    B = np.sqrt(np.power(np.subtract(C[i, 0], C[j, 0]), 2) +
                                         np.power(np.subtract(C[i, 1], C[j, 1]), 2) +
                                         np.power(np.subtract(C[i, 2], C[j, 2]), 2))
                    Binv = np.divide(1., B)
                    D[i, j] = Binv
                    D[j, i] = Binv
        xyz_inv_matrices_adni.append(D)
    return full_orig_matrices_adni, xyz_inv_matrices_adni, np.array(target_long), id_list, diagnosis, names


def convert(data, size=68, mode='vec2mat'):  # diag=0,
    '''
    Convert data from upper triangle vector to square matrix or vice versa
    depending on mode.
    INPUT : 
    data - vector or square matrix depending on mode
    size - preffered square matrix size (given by formula :
           (1+sqrt(1+8k)/2, where k = len(data), when data is vector)
    diag - how to fill diagonal for vec2mat mode
    mode - possible values 'vec2mat', 'mat2vec'
    OUTPUT : 
    square matrix or 1D vector 
    EXAMPLE :
    a = np.array([[1,2,3],[2,1,4],[3,4,1]])
    vec_x = convert(a, mode='mat2vec')
    print(vec_x)
    >>> array([2, 3, 4])
    convert(vec_x, size = 3, diag = 1, mode = vec2mat)
    >>> matrix([[1, 2, 3],
                [2, 1, 4],
                [3, 4, 1]], dtype=int64)
    '''

    if mode == 'mat2vec':

        mat = data.copy()
        rows, cols = np.triu_indices(data.shape[0], k=0)
        vec = mat[rows, cols]

        return vec

    elif mode == 'vec2mat':

        vec = data.copy()
        rows, cols = np.triu_indices(size, k=0)
        mat = csr_matrix((vec, (rows, cols)), shape=(size, size)).todense()
        mat = mat + mat.T  # symmetric matrix
        np.fill_diagonal(mat, np.diag(mat) / 2)

    return mat


def loaS_xdni(path):
    '''
    Simple script to import ADNI data set

    Данный набор данных содержит 807 снимков для 255 пациентов,
    каждому снимку поставлен в соответствие граф размера с 68 вершинами,
    метка класса (EMCI, Normal, AD, LMCI, SMC), а так же метка пациентов 
    (так как для каждого пациента есть несколько снимков,
    метки класса для одного пациента одинаковы для всех его снимков)

    IMPUT :

    path - this folder should contain 2 folders ("matrices" and "adni2_centers")
           and 1 excel file ("ADNI2_Master_Subject_List.xls")

    OUTPUT : 

    data - numpy array of shape #subjects x #nodes x #nodes
    target - numpy array containing target variable
    data_copy - pandas dataframe containing 
                subject_id, 
                scan_id (multiple scans for some patients),
                adjacency matrices (data converted to vectors)
                target (diagnosis - AD, Normal, EMCI, LMCI, SMC)

    EXAMPLE : 

    path = 'notebooks to sort/connectomics/ADNI/Data'
    data, target, info = loaS_xdni(path)


    TODO : 

    Add physical nodes position
    '''

    path_matrices = path + '/matrices/'
    path_subject_id = path + '/ADNI2_Master_Subject_List.xls'

    all_matrices = pd.DataFrame(columns=['subject_id_file', 'subject_id', 'scan_id', 'matrix', 'target'])

    # import data
    for foldername in sorted(os.listdir(path_matrices)):
        for filename in sorted(os.listdir(path_matrices + foldername)):
            if 'NORM' not in filename:
                mat = np.genfromtxt(path_matrices + foldername + '/' + filename)
                subject_id_file = foldername
                subject_id = foldername[:-2]
                scan_id = foldername[-1:]

                # ADNI data have zeros on 3 and 38 row and column
                mat = np.delete(mat, [3, 38], 1)
                mat = np.delete(mat, [3, 38], 0)

                subject_data = convert(mat, mode='mat2vec')
                single_subject = pd.DataFrame(data=[[subject_id_file, subject_id, scan_id, subject_data, np.nan]],
                                              columns=['subject_id_file', 'subject_id', 'scan_id', 'matrix', 'target'])
                all_matrices = all_matrices.append(single_subject)

    all_matrices.index = all_matrices.subject_id_file
    subject_data = pd.read_excel(path_subject_id, sheetname='Subject List')
    subject_id_names = np.array(all_matrices['subject_id_file'])

    # importing target variables
    for name in subject_id_names:
        smth = subject_data.loc[subject_data['Subject ID'] == name[:-2]]['DX Group'].dropna()
        un_smth = np.unique(smth)
        try:
            val = un_smth[0].replace(' ', '')
            all_matrices.set_value(name, 'target', val)
        except:
            pass

    # drop objects without any target
    all_matrices.dropna(inplace=True)
    data_copy = all_matrices.copy()

    temp = data_copy['matrix']

    data_vectors = np.zeros((807, 2346))
    data = np.zeros((807, 68, 68))

    for idx, vec in enumerate(temp):
        data_vectors[idx] = vec
        data[idx] = convert(vec)

    target = all_matrices.target.values
    patients_ids = data_copy.subject_id.values

    print('ADNI data shape                   :', data.shape,
         '\nADNI target variable shape        :', target.shape,
         '\nADNI number of unique patients    :', data_copy.subject_id.unique().shape)
    return data, target, data_copy


def _return_unique_labels(full_labels, full_target):
    df = pd.DataFrame(data=np.concatenate((full_target[:, np.newaxis], full_labels[:, np.newaxis]), axis=1))
    df.drop_duplicates(inplace=True)

    return df[1].values, df[0].values


def produce_dataset(data_path):
    data, target_original, info = loaS_xdni(data_path)

    orig_xd = np.where(target_original == 'AD')[0]
    orig_lmci = np.where(target_original == 'LMCI')[0]
    orig_emci = np.where(target_original == 'EMCI')[0]
    orig_nc = np.where(target_original == 'Normal')[0]
    sorted_indexes = np.concatenate((orig_xd, orig_lmci, orig_emci, orig_nc))

    needed_data = data[sorted_indexes].copy()
    for mat in needed_data:
        np.fill_diagonal(mat, 0)

    disconnected = np.where(np.sum(needed_data, 1) == 0)[0]
    clean_data = scipy.delete(needed_data, disconnected, 0)

    target_needed = target_original[sorted_indexes]
    target_clean = scipy.delete(target_needed, disconnected)
    # print target_clean.shape
    needed_users = info["subject_id"].values[sorted_indexes]
    user_id = scipy.delete(needed_users, disconnected)
    # print user_id.shape

    ad = (target_clean == 'AD').sum()
    lmci = (target_clean == 'LMCI').sum()
    emci = (target_clean == 'EMCI').sum()
    nc = (target_clean == 'Normal').sum()

    groups = np.concatenate((user_id[target_clean == 'AD'],
                             user_id[target_clean == 'LMCI'],
                             user_id[target_clean == 'EMCI'],
                             user_id[target_clean == 'Normal']))

    encoder = LabelEncoder()
    full_groups = encoder.fit_transform(groups)

    n = ad + lmci + emci + nc

    idx_xd_lmci = np.arange(ad + lmci)
    idx_xd_emci = np.concatenate((np.arange(ad), np.arange(ad + lmci, ad + lmci + emci)))
    idx_xd_nc = np.concatenate((np.arange(ad), np.arange(n - nc, n)))
    idx_lmci_emci = np.arange(ad, ad + lmci + emci)
    idx_lmci_nc = np.concatenate((np.arange(ad, ad + lmci), np.arange(n - nc, n)))
    idx_emci_nc = np.arange(ad + lmci, n)
    print('AD {}, LMCI {}, EMCI {}, NC {}'.format(ad, lmci, emci, nc))

    full_target_vector_xd_lmci = np.array([1] * ad + [0] * lmci)
    full_target_vector_xd_emci = np.array([1] * ad + [0] * emci)
    full_target_vector_xd_nc = np.array([1] * ad + [0] * nc)
    full_target_vector_lmci_emci = np.array([1] * lmci + [0] * emci)
    full_target_vector_lmci_nc = np.array([1] * lmci + [0] * nc)
    full_target_vector_emci_nc = np.array([1] * emci + [0] * nc)

    full_labels_xd_lmci = full_groups[idx_xd_lmci]
    full_labels_xd_emci = full_groups[idx_xd_emci]
    full_labels_xd_nc = full_groups[idx_xd_nc]
    full_labels_lmci_emci = full_groups[idx_lmci_emci]
    full_labels_lmci_nc = full_groups[idx_lmci_nc]
    full_labels_emci_nc = full_groups[idx_emci_nc]

    tasks = [idx_xd_lmci, idx_xd_emci, idx_xd_nc, idx_lmci_emci, idx_lmci_nc, idx_emci_nc]

    names_unique_xd_lmci, target_unique_xd_lmci = _return_unique_labels(full_labels_xd_lmci,
                                                                      full_target_vector_xd_lmci)
    names_unique_xd_emci, target_unique_xd_emci = _return_unique_labels(full_labels_xd_emci,
                                                    full_target_vector_xd_emci)
    names_unique_xd_nc, target_unique_xd_nc = _return_unique_labels(full_labels_xd_nc,
                                                  full_target_vector_xd_nc)
    names_unique_lmci_emci, target_unique_lmci_emci = _return_unique_labels(full_labels_lmci_emci,
                                                                         full_target_vector_lmci_emci)
    names_unique_lmci_nc, target_unique_lmci_nc = _return_unique_labels(full_labels_lmci_nc,
                                                                      full_target_vector_lmci_nc)
    names_unique_emci_nc, target_unique_emci_nc = _return_unique_labels(full_labels_emci_nc,
                                                                      full_target_vector_emci_nc)

    full_target_vectors = [full_target_vector_xd_lmci, full_target_vector_xd_emci,
                           full_target_vector_xd_nc, full_target_vector_lmci_emci,
                           full_target_vector_lmci_nc, full_target_vector_emci_nc]

    full_labels = [full_labels_xd_lmci, full_labels_xd_emci,
                   full_labels_xd_nc, full_labels_lmci_emci,
                   full_labels_lmci_nc, full_labels_emci_nc]

    names_unique = [names_unique_xd_lmci, names_unique_xd_emci,
                    names_unique_xd_nc, names_unique_lmci_emci,
                    names_unique_lmci_nc, names_unique_emci_nc]

    target_unique = [target_unique_xd_lmci, target_unique_xd_emci,
                     target_unique_xd_nc, target_unique_lmci_emci,
                     target_unique_lmci_nc, target_unique_emci_nc]

    return {'matrices': clean_data, 'target': full_target_vectors, 'labels': full_labels,
            'target_unique': target_unique, 'names': names_unique, 'tasks': tasks}
