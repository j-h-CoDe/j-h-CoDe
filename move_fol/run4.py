#!/usr/bin/env python
# coding: utf-8

# In[30]:

import subprocess
import os

# 환경 변수와 모듈 설정
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 모듈 로드 (환경에 따라 다를 수 있음)
#subprocess.run(["module", "load", "cuda-11.2"])
#subprocess.run(["module", "load", "icc/icc-21.1"])
#subprocess.run(["module", "load", "icc/openmpi-4.1.0"])
#subprocess.run(["module", "load", "icc/fftw-3.3.8"])
#subprocess.run(["module", "load", "icc/vasp-6.1.1-gpu"])

# 파이썬 스크립트 실행
subprocess.run(["python", "2023_GCN_Regression_Model_4.py"])

# stdout과 stderr를 파일로 리다이렉션하려면:
with open("AINB_4", "w") as out, open("4.err", "w") as err:
    subprocess.run(["python", "2023_GCN_Regression_Model_4.py"], stdout=out, stderr=err)
