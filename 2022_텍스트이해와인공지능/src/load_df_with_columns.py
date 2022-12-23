import numpy as np
import pandas as pd
import os
from src.load_df import *

def make_df_with_columns(table_name, column_list, ymd_standard,
                         data_path, filelist, ymd_seq_dict):
    # 기준날짜보다 빠른 날짜의 기록들은 날려버리기 위함
    temp = make_df(table_name,filelist,data_path)
    column_list = list(np.intersect1d(column_list,temp.columns.tolist()+['PT_SBST_NO']))
    
    
    table_name_abb = table_name
#     table_name_abb = abb(table_name,filelist)
    if table_name_abb in ymd_seq_dict.keys():
        
        # 기준날짜(대장암 진단 날짜)보다 빠른 날짜의 기록들은 날려버림
        ymd = ymd_seq_dict[table_name_abb]['ymd']
        temp_merged = pd.merge(ymd_standard,temp,how='right')
        ymd_standard_colname = ymd_standard.columns[1]
        cond = temp_merged[ymd_standard_colname]<=temp_merged[ymd]
        temp_merged_cond = temp_merged[cond]
        temp = temp_merged_cond.drop(ymd_standard_colname,axis=1)
        
        if ymd in column_list:
            ymd_list = []
        else:
            ymd_list = [ymd]
    
        if 'seq' in ymd_seq_dict[table_name_abb].keys():
            seq = ymd_seq_dict[table_name_abb]['seq']
            ymd_seq = ymd_list + [seq]
        else:
            ymd_seq = ymd_list 
        result = temp[['PT_SBST_NO']+ ymd_seq + column_list]
    else:
        result = temp[['PT_SBST_NO'] + column_list]
        
    print(f'{table_name}에서 읽어들인 컬럼들 :  {result.columns.tolist()}')    
    return result