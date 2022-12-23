import pandas
from src.load_df import *

# dummy화를 하되 같은 컬럼이 나오도록 조정
def dummy(Series, prefix, categories ,fill_null=False):
    if len(np.intersect1d(Series.values, categories))>=1:
        result = pd.DataFrame(columns = list(map(lambda x:prefix+'_'+str(x), categories)))
        real_data = pd.get_dummies(Series,prefix=prefix)
        val_columns = np.intersect1d(real_data.columns.tolist(),result.columns.tolist())
        result[val_columns] = real_data[val_columns]
        # null값은 그대로 가져가도록 (fillnull=True하면 기존 null값들이 모두 0으로 바뀜)
        if fill_null==False:
            result[Series.isnull()]=np.nan
        null_cols = result.notnull().sum()[result.notnull().sum()==0].index.tolist()
        result.loc[Series.notnull(), null_cols] = 0
        return result
    else:
        result = pd.DataFrame(np.full(shape=(Series.shape[0], len(categories)), fill_value=np.nan), columns=categories)
        return result
    
# pivot을 하되 같은 컬럼이 나오도록 조정
def pivot(tbl, index, columns, values, remain_columns):
    pivoted = tbl.pivot(index=index, columns=columns, values=values)
    pivoted_cols = pivoted.columns.tolist()
    non_exist = np.setdiff1d(remain_columns,pivoted_cols)
    pivoted[non_exist] = np.nan
    return pivoted[remain_columns]

# numeric으로 변환 가능한 값과 불가능한 값 구분
def is_numeric(x):
    try:
        float(x)
        return True
    except:
        return False

# 1,2,기타,null 값만 존재하는 컬럼에 대해 0,1,null,null로 변환
def binary(x):
    if x==1:
        y=0
    elif x==2:
        y=1
    else:
        y=np.nan
    return y

# 이상한 문자가 끼어있을시 숫자형으로 대체(map과 함께 사용)
def erase(reps, asnumeric):
    def erase_str(string):
        string = str(string)
        for rep in reps:
            string = string.replace(rep,'')
        if string=='':
            string=0
        if asnumeric==True:
            try:
                return float(string)
            except:
                return string
        else:
            return string
    return erase_str
    
# 한 사람당 여러 행이 있을때 시간순으로 가장 첫번째, 혹은 가장 마지막번째만 남김    
def SEQ_remain(df, table_name, SEQ, filelist, order_basis):
    table_name_abb = table_name
#     table_name_abb = abb(table_name,filelist)
    
    if table_name_abb in order_basis.keys():
                
        ymd = order_basis[table_name_abb]['ymd']
        if 'seq' in order_basis[table_name_abb].keys():
            seq = order_basis[table_name_abb]['seq']
            is_seq = True
        else:
            seq = []
            is_seq = False
        # SEQ에 대한 그루핑 기준은 환자 번호 + 날짜까지 포함하도록
        if SEQ=='first':
            ymd_SEQ = df.groupby('PT_SBST_NO')[ymd].transform(min)
            if is_seq:
                seq_SEQ = df.groupby(['PT_SBST_NO',ymd])[seq].transform(min)
        elif SEQ=='last':
            ymd_SEQ = df.groupby('PT_SBST_NO')[ymd].transform(max)
            if is_seq:
                seq_SEQ = df.groupby(['PT_SBST_NO',ymd])[seq].transform(max)
        else:
            raise ValueError('You can use only "first" or "last" at SEQ parameter')
        if is_seq:
            df_remain = df[(df[seq] == seq_SEQ) & (df[ymd] == ymd_SEQ)]
        else:
            df_remain = df[(df[ymd] == ymd_SEQ)]
        return df_remain
    else:
        return df
    
