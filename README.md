## Contents

<ul>
  <li style="list-style-type:circle;"> Dataset
  <li style="list-style-type:circle;"> Codes
</ul>


## Environment

The codes of our VCGAE++ are implemented under the following development environment:

<ul>
  <li style="list-style-type:circle;">python=3.6.9</li>
  <li style="list-style-type:circle;">tensorflow=1.15.3</li>
  <li style="list-style-type:circle;">numpy=1.18.5</li>
  <li style="list-style-type:circle;">scipy=1.5.4</li>
</ul>




## How to Run the Codes

<ul>
  <li style="list-style-type:circle;">JD</li>
</ul>


```python
cd VCGAE-b3
cd Code
python VCGAE.py --dataset JD  --n=10690  --m=13465 --tst_file /tst_buy --layer_size=[100,100,100]   --lr=0.001  --node_dropout_flag=1  --node_dropout=[0.1] --mess_dropout=[0.1]   --tradeOff=0.1  --tradeOff_cr=0.1 --Ks=[5,10,15] --gpu_id=4 --tradeOff_ssl=0.1 --ssl_temp=0.2

```

<ul>
  <li style="list-style-type:circle;">Tmall</li>
</ul>


```python
cd VCGAE-b3
cd Code
python VCGAE.py --dataset=Tmall --n=17202  --m=16177  --tst_file /tst_buy  --layer_size=[100,100,100,100]   --lr=0.001 --node_dropout_flag=1  --node_dropout=[0.1] --mess_dropout=[0.1]   --tradeOff=0.01  --tradeOff_cr=0.1 --Ks=[5,10,15] --gpu_id=2 --tradeOff_ssl=0.01 --ssl_temp=0.2
```

<ul>
  <li style="list-style-type:circle;">UB</li>
</ul>


```python
cd VCGAE-b4
cd Code
python VCGAE.py --dataset=UB --n=20443  --m=30947  --tst_file /tst_buy  --layer_size=[100,100,100,100] --lr=0.001    --node_dropout_flag=1  --node_dropout=[0.1]   --mess_dropout=[0.5] --tradeOff=1 --tradeOff_cr=1  --Ks=[5,10,15] --gpu_id=6 --tradeOff_ssl=0.01 --ssl_temp=0.2
```

<ul>
  <li style="list-style-type:circle;">Kuai</li>
</ul>


```python
cd VCGAE-b4
cd Code
python VCGAE.py --dataset Kuai --n=6083  --m=29758  --tst_file /tst_buy  --layer_size=[100,100,100,100] --lr=0.001    --node_dropout_flag=1  --node_dropout=[0.1]   --mess_dropout=[0.1]   --tradeOff=1 --tradeOff_cr=1  --Ks=[5,10,15] --gpu_id=1 --tradeOff_ssl=0.1 --ssl_temp=0.6
```



