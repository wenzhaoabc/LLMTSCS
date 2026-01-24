# 生成高车流量CityFlow文件

使用`LLMTSCS/converts/gen_flow.py`脚本生成高车流量的CityFlow文件.

即利用现有车流数据的转向比例，生成更高车流量的CityFlow车流文件。

# 转换CityFlow和SUMO的路网文件及车流量文件

使用以下方式进行随机车流的生成，但是由于CityFlow生成SUMO路网时不包含转向关系，导致车流route报错。

CF生成路网 -> 转为SUMO路网 -> 生成SUMO车流 -> 转为CityFlow车流*
                ↓
            转为CityFlow路网*


## 生成CityFlow的路网文件

使用CityFlow本身提供的`tools/generator/generate_grid_scenario.py`文件生成路网文件

```sh
export SUMO_HOME="/usr/share/sumo/"
```

```sh
python /home/wen/tr/LLMTSCS/CityFlow/tools/generator/generate_grid_scenario.py 4 4 \
    --rowDistance 800 \
    --columnDistance 800 \
    --laneMaxSpeed 11.11 \
    --vehMaxSpeed 11.11 \
    --vehHeadwayTime 2.0 \
    --interval 1.0 \
    --turn \
    --dir . \
    --roadnetFile cf_synthetic_roadnet.json \
    --flowFile cf_example_flow.json
```

## 将CityFlow的路网文件转换为SUMO的路网文件

```sh
python convert.py --typ c2s --f net --or_cityflownet cf_synthetic_roadnet.json --sumonet sumo_4_4_roadnet.xml
```

## 随机生成SUMO的车流文件

使用SUMO自带的`randomTrips.py`脚本生成车流文件

**注意事项**：
- 使用 `--min-distance` 确保路径足够长（CityFlow需要至少2条道路）
- 使用 `--intermediate` 增加路径复杂度，避免过短路径
- 确保 `insertion-rate` 和 `binomial` 参数合理，避免车流过密

1. 生成trips文件
```sh
python $SUMO_HOME/tools/randomTrips.py \
  -n sumo_4_4_roadnet.xml \
  -o trips_poisson_car.xml \
  -b 0 \
  -e 3600 \
  --insertion-rate 8000 \
  --binomial 10 \
  --vehicle-class passenger \
  --min-distance 2000 \
  --intermediate 2 \
  --validate
```

2. 生成rou文件
```sh
duarouter \
  -n sumo_4_4_roadnet.xml \
  --route-files trips_poisson_car.xml \
  -o routes_poisson_car.xml \
  --ignore-errors
```

## 将SUMO的车流量文件转为CityFlow的车流量文件

**重要**：确保 `--cityflownet` 参数使用的是从同一个SUMO路网转换来的CityFlow路网文件！
  
```sh
python convert.py --typ s2c --f traffic \
  --or_sumonet sumo_4_4_roadnet.xml \
  --cityflownet cf_synthetic_roadnet.json \
  --or_sumotraffic routes_poisson_car.xml \
  --cityflowtraffic cf_synthetic_flow_converted.json \
  --sumocfg 4_4_8000.sumo.cfg
```

最终得到`cf_synthetic_flow_converted.json`文件，即为转换后的CityFlow车流量文件。

## 将SUMO的路网文件转为CityFlow的路网文件

```sh
python convert.py --typ s2c --f net \
  --or_sumonet sumo_4_4_roadnet.xml \
  --cityflownet cf_sumo_converted_roadnet.json
```


### 常见问题

**问题1：CityFlow显示 "Invalid route" 警告**
- **原因**：路径不连通、道路不存在、路径太短
- **解决**：
  1. 确保使用相同源的路网文件转换
  2. 在生成SUMO trips时添加 `--min-distance 2000 --intermediate 2`
  3. 检查转换后的路网是否保留了所有道路连接