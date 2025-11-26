import pandas as pd
from scipy import stats
import numpy as np

def explanatory_analysis(charges_data_path='./data/charges_data.csv', personal_data_path='./data/personal_data.csv', plan_data_path='./data/plan_data.csv'):
    # write you solution here
    df_c = pd.read_csv(charges_data_path)
    
    col = df_c['monthlyCharges'].dropna()
    lower = col.quantile(0.1)
    upper = col.quantile(0.9)
    
    trimmed = col[(col>lower)&(col<upper)]
    # trimmed_mean = int(round(trimmed.mean(),0))
    trimmed_mean = trimmed.mean()
    
    df_c['monthlyCharges'].fillna(trimmed_mean,inplace=True)
    df_c['totalCharges'].fillna(df_c['monthlyCharges']*df_c['tenure'])

    bins = [0,24,48,60, np.inf]
    labels = ['group1','group2','group3','group4']
    df_c['tenure_bin'] = pd.cut(df_c['tenure'],bins=bins,labels=labels,right=True)
    df_c = pd.get_dummies(df_c, columns = ['tenure_bin'])


    churn_rate = int(round((df_c['churn']=='Yes').mean()*100,0))

    df_p1 = pd.read_csv(personal_data_path)

    merge_1 = pd.merge(df_c,df_p1,how='inner',on=['customerID'])
    df_p2 = pd.read_csv(plan_data_path)
    merge_2 = pd.merge(merge_1,df_p2,how='right',on=['customerID'])
    sixty_rate = int(round((merge_2['age']>60).mean()*100,0))
    
    count = merge_2.groupby('internetService').size().to_dict()
    results = {
    "monthly_charges_mean":trimmed_mean,
    "charges_data_updated":df_c,
    "churn_pct":churn_rate,
    "data_merged":merge_2,
    "pct_age_above_60":sixty_rate,
    "internet_service_counts":count
    }
    return results


"""
    1. 읽기
    2. charges_data -> monthlyCharges, totalCharges 비어있음
        monthlyCharges : 양측 10% 잘라낸 평균으로 채울 것 . 정수 반올림
        totalCharges  : monthlychatges * tenure
    3. tenureBinned 만들기 (0,24] ,(24,48] , (48,60], (60,inf)   -> ex. 23 -> (1,0,0,0) 
    4. churn rate : 백분율, 정수반올림
    5. charges_data, personal data : customerID 로 join -> 둘다 있는 경우 
        그다음 plan data까지 : 여기는 plandata에 결과 join (right join)
    6. 60살 이상 rate : 백분율, 정수반올림
    7. internetservice 개수 딕셔너리
"""
print(explanatory_analysis()["monthly_charges_mean"])