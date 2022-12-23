import numpy as np
import pandas as pd
import os

def read_table(filename):
    if 'csv' in filename:
        try:
            result = pd.read_csv(filename, low_memory=False)
        except:
            result = pd.read_csv(filename,engine='python', error_bad_lines=False)
    elif 'xlsx' in filename:
        result = pd.read_excel(filename, engine='openpyxl')
#         result.columns = result.columns.str.upper()
    return result

def slicing_table_name(full_name):
    organ = ['_CLRC_','_LUNG_','_BRST_','_LVER_']
    return ''.join(list(map(lambda x:full_name[full_name.find(x)+1:] if full_name.find(x)>0 else '',organ))).replace('.csv','').replace('.xlsx','')

def slicing_table_name_abb(full_name):
    organs = ['CLRC','LUNG','BRST','LVER']
    fileNM_list = full_name.split('_')
    find_idx = lambda x: fileNM_list.index(x) if x in fileNM_list else 0
    idx = sum([find_idx(organ) for organ in organs])
    return '_'.join(fileNM_list[idx:idx+3]).replace('.csv','').replace('.xlsx','')

def make_filelist(data_path):
    filelist = pd.Series(os.listdir(data_path),name='filename')
#     filelist_sliced = filelist.map(slicing_table_name)
    filelist_sliced = filelist.map(slicing_table_name_abb)
    filelist_df = pd.DataFrame()
    filelist_df['filename'] = filelist
    filelist_df['table_name'] = filelist_sliced
#     filelist_df['table_name_abb'] = filelist_sliced_abb
    return filelist_df

def abb(table_name, filelist):
    return filelist.loc[filelist['table_name'] == table_name,'table_name_abb'].values[0]

def make_df(table_name,filelist, data_path):
    cond = filelist['table_name'] == table_name
    filename = filelist[cond]['filename'].values[0]
    df = read_table(os.path.join(data_path,filename))
    return df

def make_ymd_seq_dict(file):
    valid = pd.read_excel(file,engine='openpyxl')
    valid = valid[['테이블 영문명','컬럼 영문명','테이블 한글명','컬럼 한글명','처리','순번구분']]
    tbl_col_df = valid.loc[valid['순번구분'].notnull()][['테이블 영문명','순번구분','컬럼 영문명']].groupby('테이블 영문명').agg(list)[['순번구분','컬럼 영문명']]
    tbl_col_se = tbl_col_df.apply(lambda x:dict(zip(x['순번구분'],x['컬럼 영문명'])),axis=1)
    # 각 테이블별로 날짜와 순번 컬럼 -> 지정한 seq변수에 따라 첫째 기록을 남길지 마지막 기록을 남길지 선택하기 위함
    ymd_seq_dict = dict(zip(tbl_col_se.index,tbl_col_se.values))
    return ymd_seq_dict
