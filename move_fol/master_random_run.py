#!/usr/bin/env python
# coding: utf-8

# In[30]:
import os
import sys
import pandas as pd
import shutil
import time
import numpy as np


from scipy.stats import norm
from scipy.special import ndtr
from modAL.utils.selection import multi_argmax
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit

import subprocess
import glob
import re



def read_csv_files(dir_):
    """
    폴더 내 모든 CSV 파일을 읽어서 데이터프레임으로 반환합니다.
    """
    file_paths = glob.glob(dir_ + "/*.csv")
    dfs = []
    for file_path in file_paths:
        dfs.append(pd.read_csv(file_path))
    return pd.concat(dfs, axis=1)

def update_filenames(filenames, old_iter, new_iter):
    """
    파일 이름을 변경합니다.
    """
    return [filename.replace(f"Iter{old_iter}", f"Iter{new_iter}") for filename in filenames]


def run_models(n_models):
    """
    모델을 실행합니다.
    """
    for j in range(n_models):
        # run0.py, run1.py, ..., run4.py와 같은 파일을 순차적으로 실행합니다.
        print(f"run{j}.py")
        subprocess.run(["python", f"run{j}.py"])

def wait_until_all_files_exist(file_paths, num_files, sleep_time):
    """
    모든 파일이 존재할 때까지 대기합니다.
    """
    while True:
        print("i'm in waiting")
        num_existing_files =  len(glob.glob(os.path.join(file_paths, "*alphafold*")))
        if num_existing_files >= num_files:
            break
        time.sleep(sleep_time)
        
def EI(mean, std, max_val, tradeoff):
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff)*ndtr(z) + std*norm.pdf(z)        

def move_file_func(f):
    dest_path = f'./Result/Iter_{f}/'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    # train_set 이동
    shutil.copy2('train.csv',
                'Before/train' + str(f) + '.csv')
    # test_set 이동
    shutil.copy2('test.csv',
                'Before/test' + str(f) + '.csv')

    for i in range(0, 5):
        shutil.copy2(f'./AINB_{i}', dest_path)
        shutil.copy2(f'./{i}.err', dest_path)
        shutil.copy2(f'./checkpoint_{i}.pt', dest_path)
        shutil.copy2(f'./mymodel_{i}.pt', dest_path)



def ddp(iteration):
    f=0
    while f <= iteration:
        os.makedirs( f'./Result/Iter_{f}', exist_ok=True)

        Start=time.time()
        run_models(5)
        wait_until_all_files_exist("./Result/Iter_"+str(f), 5, 5)
        total_csv = read_csv_files("./Result/Iter_"+str(f))
        total_csv = total_csv[['PDB', 'y_dynamic', '0', '1', '2', '3', '4']]
        total_csv = total_csv.loc[:, ~total_csv.columns.duplicated()]
        total_csv.dropna(inplace=True)

        std=total_csv[['0', '1', '2', '3', '4']].std(axis=1)
        mean=total_csv[['0', '1', '2', '3', '4']].mean(axis=1)
        
        total_csv["std"] = std
        total_csv["mean"]= mean 

        train_csv=pd.read_csv("train.csv")
        v_tradeoff=0.01
        num_instances=200 ## 업데이트 개수!!

        #shutil.move('/home/wnsgh1116/1_protein/AINB_Revised/train.csv','/home/wnsgh1116/1_protein/AINB_Revised/Before/train'+str(f)+'.csv')
        move_file_func(f)             #train_set 이동
        
        
        ei=EI(total_csv["mean"],
                total_csv["std"],
                np.max(train_csv["y_dynamic"]),
                tradeoff=v_tradeoff)
        total_csv["EI"] = ei

        query_idx = multi_argmax(ei, n_instances=num_instances)
        query_pcodid = total_csv['PDB'].iloc[query_idx]   

        new_train = total_csv.sample(n=num_instances)
        total_csv = total_csv.drop(new_train.index) 
        train_csv=pd.concat([train_csv, new_train],axis=0)
        
        train_csv=train_csv.reset_index(drop=True)
        total_csv=total_csv.reset_index(drop=True)
        
        #new_train=total_csv.iloc[query_pcodid.index]
        #total_csv=total_csv.drop(query_pcodid.index)
        #total_csv=total_csv.reset_index(drop=True)
        #train_csv=pd.concat([train_csv, new_train],axis=0)
        train_csv.to_csv("./train.csv",index=False)
        total_csv.to_csv("./test.csv",index=False) ## total_csv to test_csv


        for file_name in os.listdir("./"):
        # 파일 이름이 원하는 형식인지 확인합니다.
            if "2023_GCN_Regression_Model_" in file_name and file_name.endswith(".py"):
                file_path = os.path.join(".", file_name)
                
                # 파일을 읽어서 내용을 수정합니다.
                with open(file_path, "r") as file:
                    content = file.readlines()
                    last_line = content[-5]  # 마지막에서 세 번째 줄을 가져옵니다.
                    numbers = re.findall(r'\d+', last_line)  # 마지막 줄에서 숫자를 추출합니다.
                    if numbers:  # 숫자가 있다면 첫 번째 숫자를 5로 바꿉니다.
                        new_number = str(f+1) + numbers[0][1:]
                        new_last_line = re.sub(numbers[0], new_number, last_line)
                        content[-5] = new_last_line ###  content[-1] = new_last_line
                        new_content = ''.join(content)
                        # 수정된 내용을 파일에 씁니다.
                        with open(file_path, "w") as file:
                            file.write(new_content)

        End=time.time()-Start
        print(End)
        f += 1

################## run!! ##################
################## run!! ##################
# alphafold total: 20195 , iteration :100 * numinstance : 200 remain 20195 - 20000 = 195!!
# ddp(iteration) 
ddp(100)

################## run!! ##################


