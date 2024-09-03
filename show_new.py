import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import _pickle as pickle

# 特征和模型配置
model_config = {
    'Model full': {
        'feature_names': ['Age','Sex','Nation','Residence','WC','BMI','CL','EF','EH','SS','DF',
             'HR','SBP', 'DBP', 'Hb', 'WBC', 'ALT','AST',
               'SCr', 'TC', 'HDL-C','LDL-C',  
               'FLD','DM','FHTN'],
        'categorical_features': ['Sex', 'Nation', 'Residence','CL', 'EF','EH', 'SS', 'DF', 'FLD','DM','FHTN']
    },
    'Model small': {
        'feature_names': ['DBP', 'EH',  'Age', 'Nation', 'BMI', 
              'DM', 'EF', 'WC', 'SS', 'HR', 
               'FLD', 'Sex', 'SBP', 'DF', 'CL'],
        'categorical_features': ['EH', 'Nation', 'DM', 'FLD', 'EF', 'SS', 'Sex', 'DF', 'CL']
    }
}

# 定义类别映射
category_mappings = {
    'FLD': {'No': 0, 'Yes': 1},
    'FHTN': {'No': 0, 'Yes': 1},
    'Residence': {'Rural': 1, 'Urban': 2},
    'Sex': {'Male': 1, 'Female': 2},
    'Nation': {'Han': 1, 'Uyghur': 2, 'Other': 3},
    'CL': {'Illiterate or semi-literate': 1, "Primary school": 2, "Junior middle school": 3, "Senior middle school": 4, 'College degree and above': 5},
    'EF': {"Never": 1, "Occasionally": 2, "Often": 3},
    'EH': {"Meat and vegetable balance": 1, "Meat based": 2, "Vegetarian based": 3},
    'SS': {"Never": 1, "Smoking": 2, "Quit smoking": 3},
    'DF': {"Never": 1, "Occasionally": 2, "Often": 3},
    'DM': {'No': 0, 'Yes': 1},
}

#定义单位
continuous_features_unit_mappings={
    'PLT':'_{(10^9/L)}', 
    'TC':'_{(mmol/L)}',
    'TG':'_{(mmol/L)}',
    'HDL-C':'_{(mmol/L)}',
    'LDL-C':'_{(mmol/L)}',
    'SBP':'_{(mmHg)}',
    'DBP':'_{(mmHg)}',
    'AST':'_{(U/L)}', 
    'ALT':'_{(U/L)}', 
    'Age':'_{(years)}',
    'BMI':'_{(Kg/m^2)}', 
    'TBIL':'_{(μmol/L)}',
    'WBC':'_{(10^9/L)}',
    'SCr':'_{(μmol/L)}',
    'WC':'_{(cm)}',
    'HR':'_{(bpm)}', 
    'Hb':'_{(g/L)}'
}


# 加载连续特征范围和步长
with open('continuous_features_dict.pkl', 'rb') as f:
    continuous_features_dict = pickle.load(f)

# 加载模型
models = {
    'Model full': '/home/devdata/chengyl/hyptertension_cohort/catboost_regressor.pkl',
}

st.title("Risk Score Prediction")

# 模型选择框
model_name = 'Model full'

# 加载选择的模型
with open(models[model_name], 'rb') as f:
    model = pickle.load(f)
explainer = shap.TreeExplainer(model)

with open('/home/devdata/chengyl/hyptertension_cohort/model_new/iso_reg.pkl','rb') as f:
    iso_reg = pickle.load(f)

# 获取选择的模型的特征
feature_names = model_config[model_name]['feature_names']
categorical_features = model_config[model_name]['categorical_features']
continuous_features = [i for i in feature_names if i not in categorical_features]

# 左右分栏布局
col1, col2 = st.columns(2)

# 左栏：输入组件
with col1:
    st.header("Input Parameters")
    input_data = {}

    for var in categorical_features:
        options = list(category_mappings[var].keys())
        input_data[var] = st.selectbox(f"${var}$", options=options)

    for var in continuous_features:
        second_dict = continuous_features_dict[var]
        range_min = float(int(second_dict['range_min']))
        range_max = float(int(second_dict['range_max']))
        step = second_dict['step']
        continuous_features_unit = continuous_features_unit_mappings[var]
        input_data[var] = st.slider(f"${var} {continuous_features_unit}$",min_value=range_min, max_value=range_max, value=float(int(second_dict['mean'])), step=step)

# 右栏：输出结果
with col2:
    st.header("Prediction Result and Feature Importance")

    # 预测结果和特征重要性
    if st.button("Calculate"):
        # 转换输入数据为 DataFrame
        for var in categorical_features:
            input_data[var] = category_mappings[var][input_data[var]]

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_names)
        for i in categorical_features:
            input_df[i] = input_df[i].astype('category')

        st.write("Input DataFrame:")
        st.write(input_df)

        # 预测结果
        prediction = model.predict_proba(input_df)[0, 1]
        prediction = iso_reg.transform([prediction])[0]
        if model_name == 'Model full':
            if prediction < 0.15:
                risk = 'Low'
            elif prediction >= 0.8:
                risk = 'Very high'
            elif 0.3<=prediction < 0.8:
                risk = 'High'
            else:
                risk = 'Medium'
        else:
            if prediction < 0.188:
                risk = 'Low'
            elif prediction >= 0.714:
                risk = 'High'
            else:
                risk = 'Medium'

        st.write(f"Predicted Risk Score: {risk}")
        st.write(f"Prediction Value: {np.round(prediction,3)}")

        # 特征重要性
        shap_values = explainer(input_df)
        feature_importance = shap_values.values[0]
        feature_names.append('Output')
        feature_importance = np.append(feature_importance, explainer.expected_value + feature_importance.sum())
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df['Color'] = importance_df['Importance'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

        fig = px.bar(importance_df, x='Feature', y='Importance', color='Color',
                     color_discrete_map={'Positive': 'red', 'Negative': 'blue'},
                     title='Predicted Tips explained by SHAP values')
        st.plotly_chart(fig)