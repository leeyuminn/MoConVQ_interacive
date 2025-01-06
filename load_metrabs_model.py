#MeTRAbs test
import logging
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np


def load_model():
    try:
        model = tfhub.load('https://bit.ly/metrabs_s')
        print("Metrabs 모델 로드 성공! ✅")
        
    except Exception as e:
        print(f"모델 로드 실패: {e} ❌")

def main():
    load_model()

if __name__ == '__main__':
    main()
