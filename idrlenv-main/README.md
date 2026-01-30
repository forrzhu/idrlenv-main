# idrlenv
集中各种强化学习环境，包括用于聚合体的环境:
![](http://gitlab.idrl.ac.cn:88/ai4science/idrlenv/-/raw/main/figs/fourbipedal.png?ref_type=heads)

## installation
请先安装适合系统的`torch`，[链接](https://pytorch.org/get-started/previous-versions/), 例如：
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```
然后
```
git clone http://gitlab.idrl.ac.cn:88/ai4science/idrlenv
cd idrlenv
pip install -e .
pip install "gymnasium[all]"
cd test/external
pip install -e .
```

可以在`examples`目录中运行`biped_agg_test.py`检查是否能正常运行。

在`examples\rl_demo_bipedal_agg\td3`提供了TD3算法实现
```python
python td3.py --n_contents=1 --total_timesteps=100000 --save_model --capture_video
```

## to do
- [] Make `test/biped_agg_test.py` run. @Feng Zhu
- [ ] `biped_agg`环境中加入更多地形. @Feng Zhu
- [ ] Make `test/ant_agg_test.py` run. @Feng Zhu
- [ ] `ant_agg`环境中加入更多地形. @Feng Zhu