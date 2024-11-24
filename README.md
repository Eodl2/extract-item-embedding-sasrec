example:

```
python main.py --device=cpu --train_dir=default --dataset=ml-1m --state_dict_path='ml-1m_default\SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=8.maxlen=200.pth' --inference_only=true --maxlen=200 --hidden_units=8 --user_item_data='data\ml-1m.txt' --rating_data='data\ratings.dat'

```
original code:
https://github.com/pmixer/SASRec.pytorch

