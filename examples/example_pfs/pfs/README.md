# Learnware based on Prediction Future Sales (PFS) data downloaded from Kaggle
--> Data Page Link: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data
--> Code Page Link: https://www.kaggle.com/uladzimirkapeika/feature-engineering-lightgbm-top-1


# PFS任务描述
--> 目标：预测每个商店每个商品在下一个月的销量(注意：粒度为月，而不是每天)
--> 特征信息：商店所在城市信息、商品类别信息、商品价格信息、商品历史价格信息(特征工程中只使用了前三个月的历史信息然后拼接在一起)等
--> 使用的模型：XgBoost, LightGBM, LinearRegression
--> 评价指标：RMSE


* split_pfs_data.py
--> 根据Kaggle上公开的数据预处理方案处理下载的数据
--> 直接运行即可将数据根据Shop ID划分为每个商店的信息，包括：
  ----> 每个商品在每个月下的特征和目标值，存储为pandas.DataFrame格式
  ----> 字段包括：
    -- 标识信息： 'shop_id', 'item_id', 'date_block_num' (标识月份),
    -- 目标值(本月销量)： 'item_cnt_month',
    -- 城市信息： 'city_code', 'city_coord_1', 'city_coord_2', 'country_part',
    -- 商品种类信息： 'item_category_common', 'item_category_code',
    -- 该月的时间信息： 'weeknd_count', 'days_in_month',
    -- 商品是否第一次销售： 'item_first_interaction', 'shop_item_sold_before',
    -- 商品前三个月的销售量和价格信息： 
       'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3',
       'item_shop_price_avg_lag_1', 'item_shop_price_avg_lag_2', 'item_shop_price_avg_lag_3',
       'item_target_enc_lag_1', 'item_target_enc_lag_2', 'item_target_enc_lag_3',
       'item_loc_target_enc_lag_1', 'item_loc_target_enc_lag_2', 'item_loc_target_enc_lag_3', 'item_shop_target_enc_lag_1', 'item_shop_target_enc_lag_2', 'item_shop_target_enc_lag_3',
       'new_item_cat_avg_lag_1', 'new_item_cat_avg_lag_2', 'new_item_cat_avg_lag_3',
       'new_item_shop_cat_avg_lag_1', 'new_item_shop_cat_avg_lag_2', 'new_item_shop_cat_avg_lag_3',
       'item_cnt_month_lag_1_adv', 'item_cnt_month_lag_2_adv', 'item_cnt_month_lag_3_adv'
  ----> 特征： 除了'item_cnt_month'之外的列都当做特征列
  ----> 目标值： 'item_cnt_month'
  ----> 时间标识： 'data_block_num'将2013.01到2015.10月的数据标识为0-33，要预测的2015.11月数据为34
--> 存储结果分为两部分： 按照时间划分的train & val，是pandas.DataFrame格式


* pfs_cross_transfer.py
--> 在各自商店训练集上训练一个模型，然后在所有商店的测试集上测试，保存两两预测的RMSE结果，并进行分析
--> 分析包括两部分：(1) 对于一个目标商店，其余源域模型的性能均值，方差，最小值(最好的模型)，最大值，超过均值的源域数目，选择最好模型能够提升的比例等等；(2) HeatMap
--> 需要扩展的方向：(1) LightGBM, Ridge, Xgboost，以及超参数调参；(2) 特征工程去除标识信息，例如shop_id, item_id等等

* data_api.py
--> 后续封装的代码，需继续完善


* packages
--> pip install lightgbm
