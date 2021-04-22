### TimeSAN: A Time-Modulated Self-Attentive Network for Next Point-of-Interest Recommendation
#### Notes
This repo contains the implementation for the algorithm in:
```
@inproceedings{he2020timesan,
  title={TimeSAN: A Time-Modulated Self-Attentive Network for Next Point-of-Interest Recommendation},
  author={He, Jiayuan and Qi, Jianzhong and Ramamohanarao, Kotagiri},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2020},
  organization={IEEE}
}
```
### Dependency
The program is tested on Tensorflow 1.13
### Training (TimeSAN)
The following command will work for training the model on Semeval dataset. 

```
python3 main.py \
        --task_name=tokyo \
        --batch_size=128 \
        --lr=0.001 \
        --maxlen=250 \
        --hidden_units=100 \
        --time_units=100 \
        --num_blocks=2 \
        --num_epochs=1001 \
        --num_heads=1 \
```